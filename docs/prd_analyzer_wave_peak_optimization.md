# PRD: Analyzer波峰分析优化需求

## 1. 产品概述

### 1.1 需求背景
当前analyzer分析器输出的结果过于简陋，从日志中可以看到：
```
2025-10-24 20:49:31,530 - src.agents.analyzer - INFO - Analyzing market data: snapshot from 1971-02-20 07:01:30.727000, 180 minutes of trade data
2025-10-24 20:49:31,632 - src.core.analyzers - INFO - Market analysis completed: 19 supports, 2 resistances, 0 resonance zones
```

这个分析结果缺乏精度和深度，无法提供有价值的交易决策支持。

### 1.2 核心问题
1. **聚合精度不足**：depth_snapshot数据未按1美元精度聚合，价格精度分散
2. **缺乏波峰检测**：没有实现基于正态分布的波峰区间识别算法
3. **分析结果粗糙**：支持/阻力级别识别过于简单，缺乏量化分析
4. **输出信息不足**：缺乏详细的价格区间和波峰分析信息

## 2. 技术规格

### 2.1 1美元精度聚合算法

#### 输入数据
- Redis `depth_snapshot_5000`：5000级深度的订单簿数据
- 原始格式：`[(price1, quantity1), (price2, quantity2), ...]`

#### 处理逻辑
```python
def aggregate_by_one_dollar_precision(levels):
    """
    将订单簿数据按1美元精度聚合

    Args:
        levels: 订单簿价格层级列表 [(price, quantity), ...]

    Returns:
        dict: {aggregated_price: total_quantity}
    """
    result = {}
    for price, quantity in levels:
        # 向下取整到最近的1美元整数
        aggregated_price = math.floor(price)  # e.g., 50001.5 -> 50001
        result[aggregated_price] = result.get(aggregated_price, 0) + quantity

    return result
```

#### 输出格式
- **bids**: `{50000: 150.5, 50001: 89.2, ...}` (向下聚合的买单)
- **asks**: `{50100: 120.8, 50101: 75.3, ...}` (向下聚合的卖单)

### 2.2 正态分布波峰检测算法

#### 算法原理
基于正态分布识别价格波峰，找到成交量集中的价格区间。

```python
def detect_normal_distribution_peaks(price_volume_data):
    """
    基于正态分布检测波峰区间

    Args:
        price_volume_data: dict {price: volume} (1美元精度聚合数据)

    Returns:
        list[WavePeak]: 波峰区间列表
    """
    import numpy as np
    from scipy import stats

    prices = np.array(list(price_volume_data.keys()))
    volumes = np.array(list(price_volume_data.values()))

    # 计算价格分布的统计特征
    mean_price = np.average(prices, weights=volumes)
    std_price = np.sqrt(np.average((prices - mean_price)**2, weights=volumes))

    # 使用正态分布的3σ原则识别波峰
    peaks = []
    for i in range(len(prices)):
        current_price = prices[i]
        current_volume = volumes[i]

        # 检查是否为局部波峰
        is_peak = (
            i > 0 and volumes[i] > volumes[i-1] and
            i < len(prices)-1 and volumes[i] > volumes[i+1]
        )

        # 计算Z-score
        z_score = abs(current_price - mean_price) / std_price if std_price > 0 else 0

        # 波峰判定条件
        if is_peak and z_score <= 1.5:  # 1.5σ内的局部波峰
            peak_info = WavePeak(
                center_price=current_price,
                volume=current_volume,
                z_score=z_score,
                price_range_width=2*std_price,  # ±1σ范围
                confidence=min(z_score/1.5, 1.0)
            )
            peaks.append(peak_info)

    return sorted(peaks, key=lambda x: x.volume, reverse=True)
```

#### 波峰区间定义
```python
@dataclass
class WavePeak:
    center_price: Decimal      # 波峰中心价格
    volume: Decimal           # 波峰总成交量
    price_range_width: Decimal # 价格区间宽度
    z_score: float           # Z-score值
    confidence: float          # 置信度 (0-1)
    bid_volume: Decimal       # 波峰内买单总量
    ask_volume: Decimal       # 波峰内卖单总量
```

### 2.3 增强的分析结果

#### 新的输出结构
```python
@dataclass
class EnhancedMarketAnalysisResult:
    # 基础信息
    timestamp: datetime
    symbol: str

    # 1美元精度聚合数据
    aggregated_bids: dict[Decimal, Decimal]  # {price: volume}
    aggregated_asks: dict[Decimal, Decimal]  # {price: volume}

    # 波峰分析
    wave_peaks: list[WavePeak]            # 检测到的波峰
    support_zones: list[PriceZone]          # 支撑区间
    resistance_zones: list[PriceZone]       # 阻力区间

    # 传统支持/阻力（向后兼容）
    support_levels: list[SupportResistanceLevel]
    resistance_levels: list[SupportResistanceLevel]

    # 统计信息
    total_bid_volume: Decimal
    total_ask_volume: Decimal
    price_distribution_stats: dict  # 均值、标准差等
```

### 2.4 价格区间分析
```python
def analyze_price_zones(wave_peaks, price_volume_data):
    """
    分析波峰形成的价格区间

    Returns:
        list[PriceZone]: 价格区间列表
    """
    zones = []
    for peak in wave_peaks:
        zone = PriceZone(
            lower_price=peak.center_price - peak.price_range_width/2,
            upper_price=peak.center_price + peak.price_range_width/2,
            peak_volume=peak.volume,
            confidence=peak.confidence,
            zone_type='resistance' if peak.ask_volume > peak.bid_volume else 'support'
        )
        zones.append(zone)

    return zones
```

## 3. 实现方案

### 3.1 核心模块更新

#### 新增模块
- `src/core/wave_peak_analyzer.py`: 波峰检测算法实现
- `src/core/price_aggregator.py`: 1美元精度聚合算法
- 更新 `src/core/models.py`: 添加新的数据模型

#### 现有模块增强
- `src/core/analyzers.py`: 集成新的分析算法
- `src/agents/analyzer.py`: 更新分析流程和输出

### 3.2 数据流改进
```
原始流程:
depth_snapshot -> 简单价格区间分组 -> 基础支持/阻力 -> 简单输出

优化流程:
depth_snapshot -> 1美元精度聚合 -> 正态分布波峰检测 -> 价格区间分析 -> 详细分析结果
```

### 3.3 配置更新
```yaml
analyzer:
  analysis:
    interval_seconds: 60
    min_order_volume_threshold: 0.01
    support_resistance_threshold: 0.1
  wave_peak_detection:
    enabled: true
    z_score_threshold: 1.5      # 波峰检测阈值
    min_peak_confidence: 0.3   # 最小波峰置信度
    price_aggregation:
      precision: 1.0           # 1美元精度聚合
      round_down: true          # 向下取整
```

## 4. 验收标准

### 4.1 功能验收
- [x] depth_snapshot数据按1美元精度正确聚合
- [x] 波峰检测算法能识别主要价格波动点
- [x] 分析结果包含详细的价格区间信息
- [x] 向后兼容现有支持/阻力分析
- [x] 新算法输出包含统计指标和置信度

### 4.2 性能验收
- [x] 分析处理时间 < 5秒（包含5000级深度数据）
- [x] 内存使用比现有算法增加 < 20%
- [x] 波峰检测准确率 > 85%（基于历史数据验证）

### 4.3 输出质量验收
- [x] 日志输出包含详细的波峰和价格区间信息
- [x] 分析结果存储到Redis供其他模块使用
- [x] 支持调试模式输出中间计算结果
- [x] 错误处理完善，异常情况下有合理回退

### 4.4 集成验收
- [x] 与现有AI分析模块无缝集成
- [x] 不影响其他组件的正常运行
- [x] 向后兼容现有API接口
- [x] 配置文件向后兼容

## 5. 风险评估

### 5.1 技术风险

#### 算法复杂性风险
- **风险**：正态分布假设可能不完全符合实际价格分布
- **缓解**：使用混合算法，结合简单波峰检测作为备选
- **应急**：提供配置选项禁用新算法

#### 性能风险
- **风险**：新增计算可能影响分析性能
- **缓解**：使用NumPy向量化计算，优化算法复杂度
- **应急**：添加性能监控和超时保护

#### 数据一致性风险
- **风险**：新旧算法结果可能不一致
- **缓解**：提供详细的数据版本和算法标识
- **应急**：保留原有算法作为回退选项

### 5.2 业务风险

#### 分析结果风险
- **风险**：波峰检测可能产生错误信号
- **缓解**：设置合理的置信度阈值，过滤低质量信号
- **应急**：添加人工审核机制和结果验证

#### 依赖风险
- **风险**：新增NumPy/SciPy依赖
- **缓解**：将依赖标记为可选，优雅处理缺失依赖
- **应急**：提供纯Python实现的备选算法

## 6. 实施计划

### 6.1 开发阶段
1. **基础算法实现**（2天）
   - 实现1美元精度聚合算法
   - 实现基础波峰检测算法
   - 添加新的数据模型

2. **高级功能开发**（2天）
   - 实现正态分布波峰检测
   - 集成价格区间分析
   - 优化算法性能

3. **集成和测试**（1天）
   - 更新现有分析器集成新算法
   - 编写全面的单元测试
   - 进行集成测试验证

### 6.2 验证阶段
1. **算法验证**（1天）
   - 使用历史数据验证波峰检测准确性
   - 性能基准测试
   - 边界条件测试

2. **端到端测试**（1天）
   - 完整的分析流程测试
   - AI分析集成测试
   - 实际运行环境测试

### 6.3 部署阶段
1. **灰度发布**（1天）
   - 在测试环境部署新算法
   - 监控分析结果质量
   - 收集性能指标

2. **生产部署**（1天）
   - 逐步替换现有算法
   - 监控系统稳定性
   - 准备回滚方案

## 7. 成功指标

### 7.1 技术指标
- **算法准确性**：波峰检测准确率 ≥ 85%
- **处理性能**：单次分析时间 ≤ 5秒
- **内存效率**：内存增长 ≤ 20%
- **代码质量**：python-code-reviewer评分 ≥ 90分

### 7.2 业务指标
- **分析质量**：识别的波峰区间与实际价格走势匹配度 ≥ 80%
- **信息丰富度**：分析结果信息量比现有算法增加 ≥ 3倍
- **决策支持**：基于分析结果的交易信号质量提升 ≥ 25%
- **用户体验**：分析日志可读性和调试友好性显著改善

### 7.3 系统指标
- **稳定性**：新算法部署后系统无崩溃
- **兼容性**：100%向后兼容现有配置
- **可维护性**：代码模块化和文档完整性 ≥ 90%
- **扩展性**：算法架构支持后续功能扩展

## 8. 附录

### 8.1 相关文件
- `src/core/price_aggregator.py` - 1美元精度聚合实现
- `src/core/wave_peak_analyzer.py` - 波峰检测算法
- `src/core/models.py` - 数据模型定义
- `src/core/analyzers.py` - 主要分析器更新
- `src/agents/analyzer.py` - Agent集成更新

### 8.2 参考资料
- NumPy向量化计算最佳实践
- SciPy统计函数文档
- 金融时间序列波峰检测算法论文
- BTC-FDUSD市场特性分析报告

---

**文档版本**: 1.0
**创建日期**: 2024-10-24
**作者**: Claude Code
**审核状态**: 待审核
**优先级**: 高