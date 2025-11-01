"""优化版Agent Analyzer模块。

该模块提供了基于trades_window数据的AI趋势分析功能，是原agent_analyzer的优化版本。

主要组件：
- TradesAggregator: 交易数据聚合器
- DeepSeekAnalyzer: Deepseek AI分析客户端
- DiscordNotifier: Discord Webhook通知器
- ResponseFormatter: 响应格式化器
- OptimizedAgentAnalyzer: 主要分析器类

主要特性：
- 移除对5000层深度快照的依赖
- 专注于trades_window数据的聚合分析
- 集成Deepseek AI进行趋势判断
- 提供标准JSON输出格式
- 支持Discord webhook通知
- 完善的错误处理和重试机制
"""

from .deepseek_client import DeepSeekAnalyzer, TrendAnalysisResult
from .discord_notifier import DiscordNotificationManager, DiscordNotifier
from .optimized_analyzer import OptimizedAgentAnalyzer
from .response_formatter import ResponseFormatter, ResponseValidator
from .trades_aggregator import RawTradesData, TradesAggregator

__version__ = "1.0.0"
__author__ = "Claude Code Assistant"
__description__ = "Optimized Agent Analyzer for BTC-FDUSD trend analysis"

__all__ = [
    # 主要类
    "OptimizedAgentAnalyzer",

    # 数据处理
    "TradesAggregator",
    "RawTradesData",

    # AI分析
    "DeepSeekAnalyzer",
    "TrendAnalysisResult",

    # 通知服务
    "DiscordNotifier",
    "DiscordNotificationManager",

    # 响应处理
    "ResponseFormatter",
    "ResponseValidator",
]
