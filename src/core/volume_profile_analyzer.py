"""Volume Profile分析器 - 动态市场成交量分析。

这个模块提供Volume Profile分析功能：
1. 聚合24小时交易数据
2. 生成价格-成交量分布
3. 识别POC点和关键成交量区域
"""

import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any

logger = logging.getLogger(__name__)

# Volume Profile分析配置常量
DEFAULT_AGGREGATION_PRECISION = 10.0  # 默认$10聚合精度
VP_TOP_LEVELS_COUNT = 20  # 显示的顶级VP水平数量
VP_POC_THRESHOLD_PERCENTAGE = 0.8  # POC阈值百分比


class VolumeProfileAnalyzer:
    """Volume Profile分析器，用于分析动态市场成交量数据。

    这个分析器处理最近24小时的交易数据，生成Volume Profile，
    并识别主动成交密集区域和POC点。
    """

    def __init__(
        self,
        aggregation_precision: float = DEFAULT_AGGREGATION_PRECISION,
        min_volume_threshold: float = 0.1,
    ):
        """初始化Volume Profile分析器。

        Args:
            aggregation_precision: 价格聚合精度（例如：10.0表示$10精度）
            min_volume_threshold: 最小成交量阈值
        """
        if aggregation_precision <= 0:
            raise ValueError("聚合精度必须为正数")

        self.aggregation_precision = Decimal(str(aggregation_precision))
        self.min_volume_threshold = min_volume_threshold

        logger.info(
            f"Initialized VolumeProfileAnalyzer with precision=${aggregation_precision}, "
            f"min_volume_threshold={min_volume_threshold}"
        )

    def analyze_volume_profile(
        self, trades_window_data: list[Any], symbol: str = "BTCFDUSD"
    ) -> dict[str, Any]:
        """分析Volume Profile数据。

        Args:
            trades_window_data: 24小时交易窗口数据列表
            symbol: 交易符号

        Returns:
            包含Volume Profile分析结果的字典
        """
        logger.info(f"Starting Volume Profile analysis for {symbol}")

        if not trades_window_data:
            logger.warning("No trades window data available for VP analysis")
            return self._create_empty_vp_result(symbol)

        try:
            # 步骤1: 聚合交易数据到Volume Profile
            vp_data = self._aggregate_trades_to_volume_profile(trades_window_data)

            if not vp_data:
                logger.warning("No VP data generated from trades data")
                return self._create_empty_vp_result(symbol)

            # 步骤2: 分析Volume Profile特征
            vp_analysis = self._analyze_volume_profile_features(vp_data)

            # 步骤3: 识别POC点和关键区域
            poc_analysis = self._identify_poc_and_key_areas(vp_data)

            # 步骤4: 生成分析结果
            result = {
                "symbol": symbol,
                "timestamp": datetime.now(),
                "data_period_hours": 24,
                "aggregation_precision": float(self.aggregation_precision),
                "total_volume": sum(vp_data.values()),
                "price_levels_count": len(vp_data),
                "vp_data": vp_data,
                "vp_analysis": vp_analysis,
                "poc_analysis": poc_analysis,
                "status": "success",
            }

            logger.info(
                f"Volume Profile analysis completed: {len(vp_data)} price levels, "
                f"total_volume={result['total_volume']:.2f}"
            )

            return result

        except Exception as e:
            logger.error(f"Volume Profile analysis failed: {e}")
            return self._create_error_vp_result(symbol, str(e))

    def create_vp_prompt(self, vp_result: dict[str, Any]) -> str:
        """创建Volume Profile分析的DeepSeek提示词。

        Args:
            vp_result: Volume Profile分析结果

        Returns:
            格式化的提示词字符串
        """
        if vp_result.get("status") != "success":
            return "Volume Profile分析失败，无法生成分析提示。"

        symbol = vp_result.get("symbol", "UNKNOWN")
        vp_data = vp_result.get("vp_data", {})
        poc_analysis = vp_result.get("poc_analysis", {})

        # 获取顶级价格水平
        sorted_levels = sorted(vp_data.items(), key=lambda x: x[1], reverse=True)
        top_levels = sorted_levels[:VP_TOP_LEVELS_COUNT]

        # 格式化价格水平数据
        price_levels_data = []
        for price, volume in top_levels:
            price_levels_data.append(
                {
                    "价格": f"${float(price):,.2f}",
                    "成交量": f"{float(volume):.2f}",
                    "价格区间": self._get_price_range_description(float(price)),
                }
            )

        # 计算统计信息
        total_volume = vp_result.get("total_volume", 0)
        poc_price = poc_analysis.get("poc_price", 0)
        value_area_high = poc_analysis.get("value_area_high", 0)
        value_area_low = poc_analysis.get("value_area_low", 0)

        prompt = f"""请分析以下{symbol}的24小时Volume Profile数据：

**Volume Profile概况**：
- 分析周期：24小时
- 聚合精度：${vp_result.get("aggregation_precision", 0):.2f}
- 总成交量：{total_volume:.2f}
- 价格水平数量：{len(vp_data)}个
- POC点（Point of Control）：${float(poc_price):,.2f}
- 价值区间高：${float(value_area_high):,.2f}
- 价值区间低：${float(value_area_low):,.2f}

**顶级成交量价格水平（前{len(top_levels)}档）**：
{self._format_price_levels(price_levels_data)}

**分析要求**：
请基于以上Volume Profile数据，进行专业的动态市场分析，重点关注：

1. **POC点分析**：
   - POC点的市场意义和支撑/阻力作用
   - POC点周围的成交量密度分析

2. **主动成交密集区域识别**：
   - 高成交量集中的价格区间
   - 这些区域对价格走势的潜在影响
   - 适合流动性做市的候选区域

3. **市场结构分析**：
   - 成交量分布的均匀性或集中度
   - 价格发现过程中的关键节点
   - 买卖力量在动态成交中的表现

4. **流动性做市建议**：
   - 基于VP数据的最佳流动性部署区域
   - 风险控制和订单摆放策略
   - 与静态支撑阻力区域的共振分析

**输出格式**：
请按照以下结构提供分析结果：
{{
  "poc分析": {{
    "poc价格": "具体POC价格",
    "市场意义": "POC点的重要性分析",
    "支撑阻力作用": "作为支撑或阻力的强度评估"
  }},
  "成交密集区域": [
    {{
      "价格区间": "价格范围",
      "成交量": "该区间的总成交量",
      "特征": "该区域的详细特征",
      "做市适用性": "适合做市的程度评估"
    }}
  ],
  "市场结构洞察": {{
    "分布特征": "成交量分布的整体特征",
    "关键节点": "重要的价格节点分析",
    "买卖力量": "动态成交中的买卖力量对比"
  }},
  "流动性做市建议": {{
    "最佳区域": "推荐的流动性部署区域",
    "风险控制": "风险控制建议",
    "策略建议": "具体的做市策略建议"
  }}
}}

请提供客观专业的分析，避免任何直接交易建议。"""

        return prompt

    def _aggregate_trades_to_volume_profile(
        self, trades_window_data: list[Any]
    ) -> dict[Decimal, Decimal]:
        """将交易数据聚合到Volume Profile。

        Args:
            trades_window_data: 交易窗口数据

        Returns:
            聚合后的价格-成交量字典
        """
        volume_profile: defaultdict[Decimal, Decimal] = defaultdict(Decimal)

        total_processed = 0
        for minute_data in trades_window_data:
            # minute_data 是 MinuteTradeData 对象，不是 dict
            if not hasattr(minute_data, "price_levels"):
                continue

            # 处理每分钟的价格水平数据
            if not hasattr(minute_data, "price_levels") or not minute_data.price_levels:
                continue

            price_levels = minute_data.price_levels
            if not isinstance(price_levels, dict):
                continue

            for price_key, level_data in price_levels.items():
                try:
                    # 提取价格和成交量
                    if isinstance(level_data, dict):
                        volume = Decimal(str(level_data.get("total_volume", 0)))
                    else:
                        # 假设是对象格式
                        volume = Decimal(str(getattr(level_data, "total_volume", 0)))

                    if volume > 0:
                        # 按聚合精度对齐价格
                        # price_key 是字符串形式的整数价格
                        aligned_price = self._align_price_to_precision(
                            Decimal(str(price_key))
                        )
                        volume_profile[aligned_price] += Decimal(str(volume))
                        total_processed += 1

                except (ValueError, TypeError, AttributeError) as e:
                    logger.debug(f"Skipping invalid trade data: {e}")
                    continue

        logger.info(
            f"Aggregated {total_processed} trade data points to {len(volume_profile)} price levels"
        )
        return dict(volume_profile)

    def _align_price_to_precision(self, price: Decimal) -> Decimal:
        """将价格对齐到聚合精度。

        Args:
            price: 原始价格

        Returns:
            对齐后的价格
        """
        # 向下对齐到聚合精度
        aligned = (price // self.aggregation_precision) * self.aggregation_precision
        return aligned

    def _analyze_volume_profile_features(
        self, vp_data: dict[Decimal, Decimal]
    ) -> dict[str, Any]:
        """分析Volume Profile特征。

        Args:
            vp_data: Volume Profile数据

        Returns:
            VP特征分析结果
        """
        if not vp_data:
            return {}

        volumes = list(vp_data.values())
        total_volume = sum(volumes)
        max_volume = max(volumes)
        min_volume = min(volumes)
        avg_volume = total_volume / len(volumes)

        # 计算成交量分布统计
        sorted_volumes = sorted(volumes, reverse=True)
        top_10_percent_volume = sum(sorted_volumes[: max(1, len(sorted_volumes) // 10)])

        # 计算价格范围
        prices = list(vp_data.keys())
        price_range = max(prices) - min(prices)

        return {
            "total_volume": float(total_volume),
            "max_volume": float(max_volume),
            "min_volume": float(min_volume),
            "avg_volume": float(avg_volume),
            "volume_std_dev": self._calculate_std_dev(volumes, Decimal(str(avg_volume))),
            "price_range": float(price_range),
            "price_levels_count": len(vp_data),
            "top_10_percent_volume_ratio": float(top_10_percent_volume) / float(total_volume)
            if total_volume > 0
            else 0,
            "volume_concentration_score": float(max_volume) / float(avg_volume)
            if avg_volume > 0
            else 0,
        }

    def _identify_poc_and_key_areas(
        self, vp_data: dict[Decimal, Decimal]
    ) -> dict[str, Any]:
        """识别POC点和关键区域。

        Args:
            vp_data: Volume Profile数据

        Returns:
            POC和关键区域分析结果
        """
        if not vp_data:
            return {}

        # 找到POC点（最大成交量价格）
        poc_price = max(vp_data.items(), key=lambda x: x[1])
        poc_volume = poc_price[1]

        # 计算价值区间（Value Area）- 包含70%成交量的区间
        total_volume = sum(vp_data.values())
        target_volume = total_volume * Decimal("0.7")

        # 按成交量排序价格水平
        sorted_by_volume = sorted(vp_data.items(), key=lambda x: x[1], reverse=True)

        # 从POC开始向外扩展，直到达到目标成交量
        value_area_prices = [poc_price[0]]
        accumulated_volume = poc_volume

        # 向上和向下交替添加价格水平
        upward_index = 1
        downward_index = 1
        current_direction = "up"

        while accumulated_volume < target_volume and (
            upward_index < len(sorted_by_volume)
            or downward_index < len(sorted_by_volume)
        ):
            if current_direction == "up" and upward_index < len(sorted_by_volume):
                price, volume = sorted_by_volume[upward_index]
                value_area_prices.append(price)
                accumulated_volume += volume
                upward_index += 1
                current_direction = "down"
            elif current_direction == "down" and downward_index < len(sorted_by_volume):
                price, volume = sorted_by_volume[downward_index]
                value_area_prices.append(price)
                accumulated_volume += volume
                downward_index += 1
                current_direction = "up"
            else:
                break

        # 计算价值区间范围
        if value_area_prices:
            value_area_high = max(value_area_prices)
            value_area_low = min(value_area_prices)
        else:
            value_area_high = value_area_low = poc_price[0]

        # 识别其他高成交量区域
        max_volume = poc_volume
        high_volume_threshold = max_volume * Decimal("0.7")
        high_volume_areas = [
            {"price": float(price), "volume": float(volume)}
            for price, volume in vp_data.items()
            if volume >= high_volume_threshold and price != poc_price[0]
        ]

        return {
            "poc_price": float(poc_price[0]),
            "poc_volume": float(poc_volume),
            "value_area_high": float(value_area_high),
            "value_area_low": float(value_area_low),
            "value_area_range": float(value_area_high - value_area_low),
            "value_area_volume_percentage": accumulated_volume / total_volume
            if total_volume > 0
            else 0,
            "high_volume_areas": high_volume_areas[:5],  # 限制前5个
        }

    def _calculate_std_dev(self, values: list[Decimal], mean: Decimal) -> float:
        """计算标准差。

        Args:
            values: 数值列表
            mean: 平均值

        Returns:
            标准差
        """
        if len(values) < 2:
            return 0.0

        variance = sum((x - mean) ** 2 for x in values) / Decimal(str(len(values)))
        return float(variance.sqrt()) if hasattr(variance, 'sqrt') else float(variance ** Decimal('0.5'))

    def _format_price_levels(self, price_levels_data: list[dict[str, str]]) -> str:
        """格式化价格水平数据。

        Args:
            price_levels_data: 价格水平数据列表

        Returns:
            格式化的字符串
        """
        if not price_levels_data:
            return "无价格水平数据"

        formatted_lines = []
        for i, level in enumerate(price_levels_data, 1):
            line = (
                f"  {i}. 价格: {level['价格']}, "
                f"成交量: {level['成交量']}, "
                f"区间: {level['价格区间']}"
            )
            formatted_lines.append(line)

        return "\n".join(formatted_lines)

    def _get_price_range_description(self, price: float) -> str:
        """获取价格区间描述。

        Args:
            price: 价格值

        Returns:
            价格区间描述
        """
        if price >= 100000:
            return "十万价位区间"
        elif price >= 10000:
            return "万价位区间"
        elif price >= 1000:
            return "千价位区间"
        elif price >= 100:
            return "百价位区间"
        else:
            return "十价位以下"

    def _create_empty_vp_result(self, symbol: str) -> dict[str, Any]:
        """创建空的VP分析结果。

        Args:
            symbol: 交易符号

        Returns:
            空的分析结果字典
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "status": "no_data",
            "vp_data": {},
            "vp_analysis": {},
            "poc_analysis": {},
        }

    def _create_error_vp_result(
        self, symbol: str, error_message: str
    ) -> dict[str, Any]:
        """创建错误VP分析结果。

        Args:
            symbol: 交易符号
            error_message: 错误消息

        Returns:
            错误分析结果字典
        """
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            "status": "error",
            "error": error_message,
            "vp_data": {},
            "vp_analysis": {},
            "poc_analysis": {},
        }
