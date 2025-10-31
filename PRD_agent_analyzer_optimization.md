# Agent Analyzer优化模块产品需求文档(PRD)

## 1. 产品概述

### 1.1 功能背景
当前agent_analyzer模块依赖5000层深度快照数据进行市场分析，用户需要优化该模块，改为仅使用trades_window聚合数据，通过Deepseek AI进行分析，输出简化的JSON格式结果，并支持Discord webhook通知。

### 1.2 产品定位
基于历史交易数据的AI趋势分析工具，专注于：
- 简化数据输入（仅使用trades_window聚合数据）
- AI驱动的市场趋势分析
- 标准化JSON输出格式
- 实时Discord通知功能

### 1.3 核心价值
- **数据简化**：移除对5000层深度快照的依赖，降低数据复杂度
- **AI增强**：利用Deepseek大模型进行智能趋势分析
- **标准化输出**：提供统一的JSON格式响应
- **实时通知**：通过Discord webhook及时推送分析结果

## 2. 功能需求

### 2.1 核心功能

#### 2.1.1 数据聚合优化
**优先级：高**
- 移除对深度快照数据的依赖
- 专注于trades_window数据的聚合处理
- 保持现有的Volume Profile分析逻辑
- 优化数据聚合性能

#### 2.1.2 AI趋势分析
**优先级：高**
- 集成Deepseek API进行市场分析
- 支持以下趋势分类：
  - 震荡
  - 微弱看涨
  - 看涨
  - 强力看涨
  - 微弱看跌
  - 看跌
  - 强力看跌
- 提供每个档位的强度数值
- 生成分析原因说明

#### 2.1.3 标准化JSON输出
**优先级：高**
输出格式必须包含：
```json
{
  "timestamp": "2025-01-01T12:00:00Z",
  "trend": "看涨",
  "strength_levels": {
    "support_1": 0.8,
    "support_2": 0.6,
    "resistance_1": 0.7,
    "resistance_2": 0.5
  },
  "reason": "基于成交量分析和价格动量识别出的趋势方向..."
}
```

#### 2.1.4 Discord Webhook集成
**优先级：中**
- 支持发送分析结果到指定Discord webhook
- Webhook URL：`https://discord.com/api/webhooks/1433882831775338558/ZlPlAiCFA49TidMxyiapXzJ89MDkA8gYy5uOPxZRr7-orOwRa-71Hc_79Fa6D7SU2K6C`
- 提供消息发送状态反馈
- 支持发送失败重试机制

### 2.2 性能需求

#### 2.2.1 响应时间
- 数据聚合：< 5秒
- AI分析：< 30秒
- 总体响应时间：< 45秒

#### 2.2.2 可靠性
- 系统可用性：99.5%
- AI分析成功率：95%
- Discord通知成功率：90%

## 3. 技术规格

### 3.1 技术架构

#### 3.1.1 模块结构
```
agent_analyzer_optimized/
├── __init__.py
├── optimized_analyzer.py          # 优化后的主分析器
├── trades_aggregator.py           # 交易数据聚合器
├── deepseek_client.py             # Deepseek API客户端
├── discord_notifier.py            # Discord通知模块
└── response_formatter.py          # 响应格式化器
```

#### 3.1.2 数据流程
```
Redis trades_window数据
    ↓
数据聚合处理
    ↓
Deepseek AI分析
    ↓
JSON格式化
    ↓
Discord Webhook通知
```

### 3.2 API设计

#### 3.2.1 Deepseek API集成
```python
class DeepSeekAnalyzer:
    def analyze_trend(self, aggregated_data: dict) -> dict:
        """分析市场趋势"""
        pass

    def _build_analysis_prompt(self, data: dict) -> str:
        """构建分析提示词"""
        pass
```

#### 3.2.2 Discord Webhook API
```python
class DiscordNotifier:
    def send_analysis_result(self, result: dict) -> bool:
        """发送分析结果到Discord"""
        pass

    def format_discord_message(self, result: dict) -> str:
        """格式化Discord消息"""
        pass
```

### 3.3 数据模型

#### 3.3.1 聚合数据结构
```python
@dataclass
class AggregatedTradesData:
    timestamp: datetime
    symbol: str
    price_levels: dict[float, float]  # price -> volume
    total_volume: float
    trade_count: int
    price_range: tuple[float, float]
```

#### 3.3.2 分析结果结构
```python
@dataclass
class TrendAnalysisResult:
    timestamp: datetime
    trend: str  # 震荡/微弱看涨/看涨/强力看涨/微弱看跌/看跌/强力看跌
    strength_levels: dict[str, float]
    reason: str
    confidence: float
```

## 4. 实现方案

### 4.1 开发阶段

#### 阶段1：数据聚合优化（1天）
- 创建新的trades_aggregator.py模块
- 移除深度快照数据处理逻辑
- 优化Volume Profile聚合算法
- 实现数据验证机制

#### 阶段2：AI分析集成（1天）
- 实现deepseek_client.py
- 设计AI分析提示词模板
- 实现响应解析和验证
- 添加错误处理和重试机制

#### 阶段3：Discord通知（0.5天）
- 实现discord_notifier.py
- 设计Discord消息格式
- 添加发送状态跟踪
- 实现重试机制

#### 阶段4：集成测试（0.5天）
- 单元测试覆盖
- 集成测试验证
- 性能测试
- 错误场景测试

### 4.2 关键技术点

#### 4.2.1 AI提示词设计
```python
ANALYSIS_PROMPT_TEMPLATE = """
请基于以下BTC-FDUSD交易数据分析市场趋势：

数据概览：
- 时间范围：{time_range}
- 总成交量：{total_volume}
- 价格区间：{price_range}
- 成交密集点：{volume_peaks}

请分析并返回JSON格式结果：
{{
  "trend": "趋势判断(震荡/微弱看涨/看涨/强力看涨/微弱看跌/看跌/强力看跌)",
  "strength_levels": {{
    "strong_support": 0.0-1.0,
    "weak_support": 0.0-1.0,
    "strong_resistance": 0.0-1.0,
    "weak_resistance": 0.0-1.0
  }},
  "reason": "详细分析原因",
  "confidence": 0.0-1.0
}}
"""
```

#### 4.2.2 错误处理策略
- Deepseek API调用失败：重试3次，指数退避
- Discord通知失败：记录日志，不影响主流程
- 数据解析失败：返回默认趋势"震荡"

## 5. 验收标准

### 5.1 功能验收
- [ ] 成功移除深度快照数据依赖
- [ ] 正确聚合trades_window数据
- [ ] Deepseek API返回有效趋势分析
- [ ] JSON输出格式符合规范
- [ ] Discord通知功能正常工作

### 5.2 性能验收
- [ ] 数据聚合时间 < 5秒
- [ ] AI分析响应时间 < 30秒
- [ ] 总体处理时间 < 45秒
- [ ] 内存使用稳定

### 5.3 质量验收
- [ ] 单元测试覆盖率 ≥ 90%
- [ ] 代码审查评分 ≥ 90分
- [ ] 无安全漏洞
- [ ] 文档完整

## 6. 风险评估

### 6.1 技术风险

#### 6.1.1 AI API稳定性
**风险等级：中**
- 风险：Deepseek API不稳定或响应格式变化
- 缓解：实现响应格式验证，添加降级策略

#### 6.1.2 Discord通知可靠性
**风险等级：低**
- 风险：Webhook发送失败影响用户体验
- 缓解：实现重试机制，提供发送状态反馈

### 6.2 业务风险

#### 6.2.1 分析准确性
**风险等级：中**
- 风险：仅使用trades数据可能影响分析准确性
- 缓解：优化聚合算法，提供置信度评分

### 6.3 运维风险

#### 6.3.1 依赖管理
**风险等级：低**
- 风险：新增外部依赖（Discord webhook）
- 缓解：完善错误处理，提供降级方案

## 7. 监控指标

### 7.1 业务指标
- AI分析成功率
- Discord通知成功率
- 响应时间分布
- 错误率统计

### 7.2 技术指标
- CPU/内存使用率
- API调用延迟
- 数据处理吞吐量
- 系统可用性

## 8. 项目时间线

| 阶段 | 任务 | 预估时间 | 负责人 |
|------|------|----------|--------|
| 1 | 数据聚合优化 | 1天 | 开发 |
| 2 | AI分析集成 | 1天 | 开发 |
| 3 | Discord通知 | 0.5天 | 开发 |
| 4 | 测试验证 | 0.5天 | 开发+测试 |
| 5 | 代码审查 | 0.5天 | 开发 |
| **总计** | | **3.5天** | |

## 9. 成功标准

### 9.1 必须达成
- ✅ 移除5000层深度快照数据依赖
- ✅ 实现基于trades_window数据的AI分析
- ✅ 返回标准JSON格式结果
- ✅ Discord webhook通知功能正常

### 9.2 期望达成
- 🎯 响应时间 < 30秒
- 🎯 分析置信度 > 80%
- 🎯 代码质量评分 ≥ 90分

### 9.3 可选达成
- 🌟 支持多种趋势强度指标
- 🌟 提供分析历史记录
- 🌟 支持自定义Discord消息格式

---

**文档版本**: v1.0
**创建时间**: 2025-11-01
**更新时间**: 2025-11-01
**审核状态**: 待审核