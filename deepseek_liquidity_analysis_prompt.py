#!/usr/bin/env python3
"""
DeepSeek流动性分析提示词生成器
用于生成精准的AI提示词，分析70%流动性密集区域
"""

import json
from datetime import datetime
from typing import Dict, List

class LiquidityAnalysisPromptGenerator:
    """DeepSeek流动性分析提示词生成器"""

    def generate_comprehensive_analysis_prompt(self, trades_data: List[Dict], depth_data: Dict) -> str:
        """
        生成用于DeepSeek的完整流动性分析提示词

        Args:
            trades_data: 48小时1分钟级别订单流数据
            depth_data: 当前5000级深度订单簿数据

        Returns:
            str: DeepSeek提示词
        """

        prompt = f"""
# BTC-FDUSD 高频流动性密度分析任务

## 任务概述
你是一个专业的加密货币量化交易分析师，需要基于提供的48小时历史订单流数据和当前深度订单簿数据，识别出BTC-FDUSD交易对中70%流动性密集分布的关键价格区域。

## 数据提供

### 1. 48小时订单流数据样本 (共{len(trades_data)}条记录)
```json
{json.dumps(trades_data[:5], indent=2)}
```

### 2. 当前5000级深度订单簿数据
```json
{json.dumps(depth_data, indent=2)}
```

## 分析要求

### 核心目标
识别出占总成交量70%的流动性密集价格区域，这些区域具有以下特征：
- 高成交量和频繁交易
- 买卖盘深度充足
- 价格稳定性强
- 适合高频套利交易

### 分析维度

#### 1. 成交量分布分析
- 计算每个价格水平的累积成交量百分比
- 识别成交量集中度最高的价格区间
- 分析成交量-价格分布的形态特征

#### 2. 买卖压力分析
- 计算每个价格水平的买卖比率 (buy_volume / sell_volume)
- 识别买盘主导的价格区域 (buy_ratio > 0.6)
- 分析买卖压力的平衡点和转折点

#### 3. 流动性深度分析
- 基于深度订单簿计算各价格档位的挂单量
- 识别大单聚集的关键价格水平
- 分析流动性真空区域和密度区域

#### 4. 时间序列分析
- 分析不同时间段的交易活跃度
- 识别持续有流动性的时间窗口
- 分析价格在关键区域的停留时间

#### 5. 支撑阻力强度分析
- 基于历史成交数据计算支撑强度
- 识别强支撑和弱阻力区域
- 分析价格反弹和突破的概率

## 输出要求

请按照以下格式提供分析结果：

### 1. 流动性密集区域识别
```
主要流动性区域 (70%成交量覆盖):
- 中心价格: XXXXX.XX USDT
- 价格区间: [XXXXX.XX, XXXXX.XX] USDT
- 区间宽度: XXX.XX USDT (X.XX%)
- 覆盖成交量: XX.XX BTC (占总成交量70%)
- 买盘占比: XX.X%
- 卖盘占比: XX.X%
- 平均交易次数: XXXX次/分钟
```

### 2. 套利机会评估 (只做多策略)
```
做多机会分析:
- 最佳入场价格: XXXXX.XX USDT (流动性支撑位附近)
- 止损价格: XXXXX.XX USDT (支撑位下方0.5%)
- 止盈价格: XXXXX.XX USDT (区域中心上方0.2%)
- 预期胜率: XX%
- 风险收益比: 1:X.X
- 建议仓位: 总资金的X%
```

### 3. 市场结构分析
```
市场结构评估:
- 当前趋势: [上涨/下跌/震荡]
- 流动性状态: [充足/紧张/不平衡]
- 关键支撑位: XXXXX.XX USDT
- 关键阻力位: XXXXX.XX USDT
- 流动性分布: [均匀/集中/分散]
```

### 4. 交易执行建议
```
高频套利策略 (只做多):
- 入场条件: 价格回调至流动性支撑区域
- 出场条件: 达到止盈目标或触及止损
- 持仓时间: 预期5-30分钟
- 交易频率: 每日X-X次机会
- 风险控制: 单笔交易亏损不超过总资金1%
```

## 分析提示

1. **重点关注**：寻找买卖盘深度充足、成交量集中、价格稳定的关键区域
2. **安全边际**：选择的入场点应该有足够的支撑，避免在流动性真空区域交易
3. **风险控制**：严格设置止损，确保单笔交易风险可控
4. **流动性确认**：结合深度订单簿数据，确认识别的流动性区域有足够的挂单深度
5. **时效性**：重点关注最近24小时的数据，因为市场流动性特征会随时间变化

## 特别要求

1. **只做多策略**：由于风险控制要求，只寻找做多机会，不主动做空
2. **高频特征**：分析的持仓时间应该在5-30分钟范围内
3. **流动性要求**：选择的区域必须能够支持中等规模的交易执行
4. **实时性**：分析结果应该适用于当前市场状况

请基于提供的数据，进行深度分析并给出具体的交易建议。
"""

        return prompt

    def generate_signal_evaluation_prompt(self, current_signals: List[Dict], market_context: Dict) -> str:
        """
        生成信号评估提示词

        Args:
            current_signals: 当前交易信号列表
            market_context: 市场上下文信息

        Returns:
            str: 信号评估提示词
        """

        prompt = f"""
# 高频套利信号实时评估

## 当前市场状态
```json
{json.dumps(market_context, indent=2)}
```

## 待评估交易信号
```json
{json.dumps(current_signals, indent=2)}
```

## 评估要求

请评估每个交易信号的质量，并给出：
1. 信号可靠性评分 (1-10分)
2. 建议调整的仓位大小
3. 优化的入场/出场点位
4. 潜在风险提示

## 评估标准
- 流动性充足性
- 价格合理性
- 风险控制有效性
- 市场环境适配性

请针对每个信号提供详细的评估报告。
"""

        return prompt

    def generate_risk_assessment_prompt(self, position_data: Dict, market_data: Dict) -> str:
        """
        生成风险评估提示词

        Args:
            position_data: 当前持仓数据
            market_data: 市场数据

        Returns:
            str: 风险评估提示词
        """

        prompt = f"""
# 持仓风险评估与管理

## 当前持仓状况
```json
{json.dumps(position_data, indent=2)}
```

## 市场实时数据
```json
{json.dumps(market_data, indent=2)}
```

## 风险评估任务

请评估当前持仓的风险状况，并提供：
1. 市场风险等级 (低/中/高)
2. 建议的止损调整
3. 分批出场策略
4. 市场异常情况应对方案

## 风险指标
- 价格波动风险
- 流动性风险
- 市场情绪风险
- 技术面风险

请提供详细的风险管理建议。
"""

        return prompt

def main():
    """演示提示词生成"""

    # 示例数据 (实际使用时从Redis获取)
    sample_trades = [
        {
            "timestamp": "2025-10-28T17:09:31.385973",
            "price_levels": {
                "114643": {
                    "price_level": 114643.0,
                    "buy_volume": 0.08494,
                    "sell_volume": 0.04313,
                    "total_volume": 0.12807,
                    "delta": 0.04181,
                    "trade_count": 28
                }
            }
        }
    ]

    sample_depth = {
        "asks": [[114581.81, 0.0085], [114581.88, 0.00805], [114584.99, 0.0096]],
        "bids": [[114579.29, 0.00005], [114576.84, 0.01678], [114576.83, 0.0101]],
        "symbol": "BTCFDUSD",
        "timestamp": "2025-10-28T17:10:18.703877"
    }

    # 生成提示词
    generator = LiquidityAnalysisPromptGenerator()

    # 生成完整分析提示词
    analysis_prompt = generator.generate_comprehensive_analysis_prompt(sample_trades, sample_depth)

    print("=" * 80)
    print("DeepSeek 流动性分析提示词")
    print("=" * 80)
    print(analysis_prompt)

    # 保存提示词到文件
    with open('deepseek_liquidity_analysis_prompt.txt', 'w', encoding='utf-8') as f:
        f.write(analysis_prompt)

    print("\n提示词已保存到 deepseek_liquidity_analysis_prompt.txt")

if __name__ == "__main__":
    main()