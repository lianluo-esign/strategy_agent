# BTC-FDUSD短期动量策略算法 PRD文档

## 1. 功能概述

### 1.1 项目背景
基于BTC-FDUSD现货交易的trades_window数据，开发一个短期动量策略算法，用于识别市场的短期价格动量方向和强度，为高频交易提供决策支持。

### 1.2 核心目标
- 实时分析trades_window中的交易数据
- 计算短期价格动量方向（买入/卖出/中性）
- 量化动量强度（0-1之间的数值）
- 为高频交易策略提供信号输入

## 2. 技术规格

### 2.1 输入数据
```python
# trades_window 数据结构
{
    "trades": [
        {
            "timestamp": "2025-01-01T10:00:00.000Z",
            "price": 42000.50,
            "quantity": 0.1,
            "side": "buy",  # "buy" | "sell"
            "trade_id": "12345678"
        }
        # ... 更多交易记录
    ],
    "window_start": "2025-01-01T09:55:00.000Z",
    "window_end": "2025-01-01T10:00:00.000Z",
    "symbol": "BTCFDUSD"
}
```

### 2.2 输出格式
```python
# 动量分析结果
{
    "direction": "buy",        # "buy" | "sell" | "neutral"
    "strength": 0.75,          # 0.0 - 1.0
    "confidence": 0.82,        # 0.0 - 1.0
    "indicators": {
        "price_momentum": 0.68,
        "volume_momentum": 0.81,
        "order_flow_momentum": 0.76,
        "volatility_adjusted": 0.73
    },
    "timeframe": "5m",         # 分析时间窗口
    "analysis_timestamp": "2025-01-01T10:00:00.000Z"
}
```

### 2.3 算法核心指标

#### 2.3.1 价格动量指标
- **价格变化率**：计算时间窗口内的价格变化百分比
- **加权价格动量**：基于成交量的加权价格变化
- **价格趋势强度**：使用线性回归计算价格趋势斜率

#### 2.3.2 成交量动量指标
- **成交量变化率**：当前窗口与历史窗口的成交量对比
- **买卖量不平衡**：买入总量与卖出总量的比例
- **大单占比**：大额交易（>平均交易量2倍）的占比

#### 2.3.3 订单流动量指标
- **主动买卖比例**：主动买入vs主动卖出的比例
- **订单流压力**：连续同方向交易的强度
- **市场深度变化**：订单簿深度的变化趋势

#### 2.3.4 波动率调整指标
- **实现波动率**：基于价格序列计算的历史波动率
- **波动率标准化**：将动量指标按波动率调整
- **风险调整收益**：计算夏普比率的类似指标

## 3. 实现方案

### 3.1 核心算法架构
```python
class ShortTermMomentumAnalyzer:
    def __init__(self, window_size: int = 300):  # 5分钟窗口
        self.window_size = window_size
        self.price_analyzer = PriceMomentumAnalyzer()
        self.volume_analyzer = VolumeMomentumAnalyzer()
        self.flow_analyzer = OrderFlowAnalyzer()
        self.volatility_analyzer = VolatilityAnalyzer()

    def analyze_momentum(self, trades_window: TradesWindow) -> MomentumResult:
        # 1. 数据预处理
        processed_data = self.preprocess_trades_data(trades_window)

        # 2. 计算各类动量指标
        price_momentum = self.price_analyzer.calculate(processed_data)
        volume_momentum = self.volume_analyzer.calculate(processed_data)
        flow_momentum = self.flow_analyzer.calculate(processed_data)

        # 3. 波动率调整
        volatility_adjusted = self.volatility_analyzer.adjust(
            price_momentum, volume_momentum, flow_momentum
        )

        # 4. 综合评分
        final_score = self.combine_indicators(
            price_momentum, volume_momentum, flow_momentum, volatility_adjusted
        )

        # 5. 生成交易信号
        return self.generate_signal(final_score)
```

### 3.2 关键算法细节

#### 3.2.1 价格动量计算
```python
def calculate_price_momentum(self, price_series: List[float]) -> Dict[str, float]:
    """
    计算价格动量指标
    """
    # 1. 简单价格变化率
    price_change = (price_series[-1] - price_series[0]) / price_series[0]

    # 2. 线性回归趋势强度
    x = np.arange(len(price_series))
    slope, _, r_value, _, _ = stats.linregress(x, price_series)
    trend_strength = abs(slope * r_value)

    # 3. 加权价格动量（基于成交量）
    weighted_momentum = self.calculate_weighted_momentum(price_series, volumes)

    return {
        'simple_change': price_change,
        'trend_strength': trend_strength,
        'weighted_momentum': weighted_momentum
    }
```

#### 3.2.2 成交量动量计算
```python
def calculate_volume_momentum(self, trades_data: List[Trade]) -> Dict[str, float]:
    """
    计算成交量动量指标
    """
    # 1. 买卖量不平衡
    buy_volume = sum(t.quantity for t in trades_data if t.side == 'buy')
    sell_volume = sum(t.quantity for t in trades_data if t.side == 'sell')
    volume_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)

    # 2. 大单分析
    avg_trade_size = np.mean([t.quantity for t in trades_data])
    large_trades = [t for t in trades_data if t.quantity > avg_trade_size * 2]
    large_trade_ratio = len(large_trades) / len(trades_data)

    # 3. 成交量趋势
    volume_series = self.group_trades_by_interval(trades_data, interval='1m')
    volume_trend = self.calculate_volume_trend(volume_series)

    return {
        'volume_imbalance': volume_imbalance,
        'large_trade_ratio': large_trade_ratio,
        'volume_trend': volume_trend
    }
```

### 3.3 信号生成逻辑
```python
def generate_momentum_signal(self, indicators: Dict[str, float]) -> MomentumSignal:
    """
    生成动量交易信号
    """
    # 1. 计算综合动量分数
    momentum_score = (
        indicators['price_momentum'] * 0.3 +
        indicators['volume_momentum'] * 0.3 +
        indicators['order_flow_momentum'] * 0.25 +
        indicators['volatility_adjusted'] * 0.15
    )

    # 2. 确定方向
    if momentum_score > 0.15:  # 买入阈值
        direction = 'buy'
    elif momentum_score < -0.15:  # 卖出阈值
        direction = 'sell'
    else:
        direction = 'neutral'

    # 3. 计算强度和置信度
    strength = min(abs(momentum_score) * 2, 1.0)  # 归一化到0-1
    confidence = self.calculate_confidence(indicators)

    return MomentumSignal(
        direction=direction,
        strength=strength,
        confidence=confidence,
        raw_score=momentum_score,
        indicators=indicators
    )
```

## 4. 验收标准

### 4.1 功能验收标准
1. **数据处理准确性**
   - 能够正确解析trades_window数据
   - 处理缺失数据和异常值
   - 支持不同时间窗口大小

2. **算法性能要求**
   - 单次分析响应时间 < 50ms
   - 支持每秒100次以上的分析请求
   - 内存使用稳定，无明显泄漏

3. **信号质量标准**
   - 信号方向准确率 > 65%
   - 强度指标与实际价格变化相关性 > 0.7
   - 置信度能够有效预测信号成功率

### 4.2 技术验收标准
1. **代码质量**
   - 单元测试覆盖率 > 90%
   - 集成测试通过率 100%
   - 代码审查评分 > 90分

2. **系统稳定性**
   - 连续运行24小时无崩溃
   - 异常情况优雅处理
   - 日志记录完整清晰

## 5. 风险评估

### 5.1 技术风险
1. **数据质量风险**
   - 风险：trades_window数据缺失或延迟
   - 缓解：实现数据质量检查和容错机制

2. **算法过拟合风险**
   - 风险：算法在历史数据上表现良好但实盘表现差
   - 缓解：使用滚动窗口验证和参数稳定性测试

### 5.2 市场风险
1. **市场状态变化**
   - 风险：市场风格切换导致算法失效
   - 缓解：实现自适应参数调整机制

2. **流动性风险**
   - 风险：信号触发但无法及时成交
   - 缓解：结合市场深度数据评估执行可行性

## 6. 部署要求

### 6.1 系统要求
- Python 3.8+
- 内存：最小512MB，推荐1GB
- CPU：支持并发处理
- 存储：日志和历史数据存储

### 6.2 依赖库
```python
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
pytest>=6.0.0
```

## 7. 监控指标

### 7.1 性能监控
- 算法响应时间
- 内存使用情况
- CPU使用率
- 错误率统计

### 7.2 交易监控
- 信号生成频率
- 信号准确率
- 平均持仓时间
- 收益率统计

## 8. 后续优化方向

### 8.1 算法优化
1. 引入机器学习模型提升预测准确率
2. 实现多时间框架的动量分析
3. 添加市场情绪指标作为辅助判断

### 8.2 功能扩展
1. 支持多个交易对同时分析
2. 实现动量信号的回测功能
3. 添加可视化监控界面