"""动态数据波动率和成交量分析模块。

该模块提供对trades_window数据的波动率和成交量统计分析功能：
1. 价格波动率计算和分析
2. 成交量统计和趋势分析
3. 价格-成交量关系分析
4. 市场活跃度评估
"""

import logging
import math
from datetime import datetime
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VolatilityVolumeAnalyzer:
    """波动率和成交量分析器。

    专门用于分析trades_window动态数据中的波动率和成交量特征，
    为贝叶斯分析提供量化证据。
    """

    def __init__(self, min_data_points: int = 10):
        """初始化分析器。

        Args:
            min_data_points: 最少数据点要求
        """
        self.min_data_points = min_data_points

    def analyze_volatility_volume(self, dynamic_data: dict[str, Any]) -> dict[str, Any]:
        """分析动态数据的波动率和成交量特征。

        Args:
            dynamic_data: 动态交易数据字典

        Returns:
            波动率和成交量分析结果
        """
        if not dynamic_data or dynamic_data.get("status") != "success":
            return {
                "status": "error",
                "error": "Invalid dynamic data",
                "analysis_type": "volatility_volume_analysis"
            }

        minute_data_points = dynamic_data.get("minute_data_points", [])
        if len(minute_data_points) < self.min_data_points:
            return {
                "status": "insufficient_data",
                "error": f"需要至少{self.min_data_points}个数据点，当前只有{len(minute_data_points)}个",
                "analysis_type": "volatility_volume_analysis"
            }

        try:
            # 提取价格和成交量数据
            price_data = self._extract_price_series(minute_data_points)
            volume_data = self._extract_volume_series(minute_data_points)

            if not price_data or not volume_data:
                return {
                    "status": "no_valid_data",
                    "error": "无法提取有效的价格或成交量数据",
                    "analysis_type": "volatility_volume_analysis"
                }

            # 执行各项分析
            volatility_analysis = self._analyze_price_volatility(price_data)
            volume_analysis = self._analyze_trading_volume(volume_data)
            price_volume_relation = self._analyze_price_volume_relationship(price_data, volume_data)
            market_activity = self._assess_market_activity(minute_data_points, price_data, volume_data)

            # 生成综合分析摘要
            analysis_summary = self._generate_analysis_summary(
                volatility_analysis, volume_analysis, price_volume_relation, market_activity
            )

            return {
                "status": "success",
                "analysis_type": "volatility_volume_analysis",
                "timestamp": datetime.now().isoformat(),
                "data_points_analyzed": len(minute_data_points),
                "volatility_analysis": volatility_analysis,
                "volume_analysis": volume_analysis,
                "price_volume_relationship": price_volume_relation,
                "market_activity_assessment": market_activity,
                "analysis_summary": analysis_summary
            }

        except Exception as e:
            logger.error(f"波动率成交量分析失败: {e}")
            return {
                "status": "error",
                "error": str(e),
                "analysis_type": "volatility_volume_analysis"
            }

    def _extract_price_series(self, minute_data_points: list[dict[str, Any]]) -> list[float]:
        """从分钟数据中提取价格序列。

        Args:
            minute_data_points: 分钟数据点列表

        Returns:
            价格序列列表
        """
        prices = []

        for point in minute_data_points:
            price_levels = point.get("price_levels", {})
            if not price_levels:
                continue

            # 计算成交量加权平均价格
            total_volume: float = 0
            weighted_sum: float = 0

            for price_str, level_data in price_levels.items():
                try:
                    price = float(price_str)
                    volume = float(level_data.get("total_volume", 0))
                    if volume > 0:
                        weighted_sum += price * volume
                        total_volume += volume
                except (ValueError, TypeError):
                    continue

            if total_volume > 0:
                vwap = weighted_sum / total_volume
                prices.append(vwap)

        return prices

    def _extract_volume_series(self, minute_data_points: list[dict[str, Any]]) -> list[float]:
        """从分钟数据中提取成交量序列。

        Args:
            minute_data_points: 分钟数据点列表

        Returns:
            成交量序列列表
        """
        volumes = []

        for point in minute_data_points:
            price_levels = point.get("price_levels", {})
            point_volume: float = 0

            for level_data in price_levels.values():
                try:
                    volume = float(level_data.get("total_volume", 0))
                    point_volume += volume
                except (ValueError, TypeError):
                    continue

            volumes.append(point_volume)

        return volumes

    def _analyze_price_volatility(self, price_data: list[float]) -> dict[str, Any]:
        """分析价格波动率。

        Args:
            price_data: 价格数据序列

        Returns:
            价格波动率分析结果
        """
        if len(price_data) < 2:
            return {"error": "价格数据点不足"}

        # 计算价格收益率
        returns = []
        for i in range(1, len(price_data)):
            if price_data[i-1] > 0:
                ret = (price_data[i] - price_data[i-1]) / price_data[i-1]
                returns.append(ret)

        if not returns:
            return {"error": "无法计算价格收益率"}

        # 基础统计
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        variance = np.var(returns)

        # 波动率指标
        volatility_5min = std_return * math.sqrt(5)  # 5分钟波动率
        volatility_15min = std_return * math.sqrt(15)  # 15分钟波动率
        volatility_60min = std_return * math.sqrt(60)  # 60分钟波动率

        # 价格范围分析
        price_range = max(price_data) - min(price_data)
        price_range_pct = (price_range / price_data[0]) * 100 if price_data[0] > 0 else 0

        # 价格趋势强度
        price_trend = (price_data[-1] - price_data[0]) / price_data[0] if price_data[0] > 0 else 0
        trend_strength = abs(price_trend)

        # 波动率分类
        if std_return < 0.001:
            volatility_level = "极低"
        elif std_return < 0.003:
            volatility_level = "低"
        elif std_return < 0.008:
            volatility_level = "中等"
        elif std_return < 0.015:
            volatility_level = "高"
        else:
            volatility_level = "极高"

        return {
            "basic_statistics": {
                "mean_return": mean_return,
                "std_return": std_return,
                "variance": variance,
                "data_points": len(returns)
            },
            "volatility_metrics": {
                "current_volatility": std_return,
                "volatility_5min": volatility_5min,
                "volatility_15min": volatility_15min,
                "volatility_60min": volatility_60min,
                "volatility_level": volatility_level
            },
            "price_range_analysis": {
                "highest_price": max(price_data),
                "lowest_price": min(price_data),
                "price_range": price_range,
                "price_range_percentage": price_range_pct
            },
            "trend_analysis": {
                "price_trend_percent": price_trend * 100,
                "trend_direction": "上涨" if price_trend > 0 else "下跌" if price_trend < 0 else "横盘",
                "trend_strength_percent": trend_strength * 100,
                "trend_strength_level": "强" if trend_strength > 0.01 else "中" if trend_strength > 0.003 else "弱"
            }
        }

    def _analyze_trading_volume(self, volume_data: list[float]) -> dict[str, Any]:
        """分析成交量特征。

        Args:
            volume_data: 成交量数据序列

        Returns:
            成交量分析结果
        """
        if not volume_data:
            return {"error": "成交量数据为空"}

        # 基础统计
        total_volume = sum(volume_data)
        avg_volume = np.mean(volume_data)
        median_volume = np.median(volume_data)
        std_volume = np.std(volume_data)
        max_volume = max(volume_data)
        min_volume = min(volume_data)

        # 成交量趋势分析
        if len(volume_data) >= 10:
            recent_avg = np.mean(volume_data[-5:])  # 最近5个点
            earlier_avg = np.mean(volume_data[-10:-5])  # 之前5个点
            volume_trend_pct = ((recent_avg - earlier_avg) / earlier_avg * 100) if earlier_avg > 0 else 0

            volume_trend_direction = "上升" if volume_trend_pct > 10 else "下降" if volume_trend_pct < -10 else "稳定"
        else:
            volume_trend_pct = 0
            volume_trend_direction = "无法判断"

        # 成交量波动性
        volume_cv = std_volume / avg_volume if avg_volume > 0 else 0  # 变异系数
        volume_stability = "稳定" if volume_cv < 0.5 else "中等波动" if volume_cv < 1.0 else "高波动"

        # 成交量分布
        volume_distribution = {
            "q25": np.percentile(volume_data, 25),
            "q75": np.percentile(volume_data, 75),
            "iqr": np.percentile(volume_data, 75) - np.percentile(volume_data, 25)
        }

        # 成交量活跃度分类
        if avg_volume > 1000:
            activity_level = "极度活跃"
        elif avg_volume > 500:
            activity_level = "高度活跃"
        elif avg_volume > 100:
            activity_level = "活跃"
        elif avg_volume > 50:
            activity_level = "一般活跃"
        else:
            activity_level = "低活跃"

        # 成交量突增检测
        volume_spikes = []
        if len(volume_data) >= 5:
            avg_baseline = np.mean(volume_data[:-5]) if len(volume_data) > 5 else avg_volume
            spike_threshold = avg_baseline * 2.0

            for i, vol in enumerate(volume_data[-5:]):
                if vol > spike_threshold:
                    volume_spikes.append({
                        "position": len(volume_data) - 5 + i,
                        "volume": vol,
                        "spike_ratio": vol / avg_baseline if avg_baseline > 0 else 0
                    })

        return {
            "basic_statistics": {
                "total_volume": total_volume,
                "average_volume": avg_volume,
                "median_volume": median_volume,
                "std_volume": std_volume,
                "max_volume": max_volume,
                "min_volume": min_volume,
                "data_points": len(volume_data)
            },
            "trend_analysis": {
                "volume_trend_percent": volume_trend_pct,
                "trend_direction": volume_trend_direction
            },
            "volatility_analysis": {
                "coefficient_of_variation": volume_cv,
                "stability_level": volume_stability
            },
            "distribution_analysis": volume_distribution,
            "activity_assessment": {
                "activity_level": activity_level,
                "volume_spikes": volume_spikes,
                "spike_count": len(volume_spikes)
            }
        }

    def _analyze_price_volume_relationship(self, price_data: list[float], volume_data: list[float]) -> dict[str, Any]:
        """分析价格-成交量关系。

        Args:
            price_data: 价格数据序列
            volume_data: 成交量数据序列

        Returns:
            价格-成交量关系分析结果
        """
        if len(price_data) != len(volume_data) or len(price_data) < 3:
            return {"error": "数据长度不一致或数据点不足"}

        # 计算价格和成交量变化
        changes = self._calculate_price_volume_changes(price_data, volume_data)
        if not changes:
            return {"error": "无法计算价格和成交量变化"}

        price_changes, volume_changes = changes

        # 分析相关性
        correlation_analysis = self._analyze_correlation(price_changes, volume_changes)

        # 识别量价配合模式
        coordination_patterns = self._identify_coordination_patterns(price_changes, volume_changes)

        # 统计模式分布
        pattern_analysis = self._analyze_pattern_distribution(coordination_patterns)

        return {
            "correlation_analysis": correlation_analysis,
            "coordination_patterns": coordination_patterns,
            "relationship_assessment": pattern_analysis
        }

    def _calculate_price_volume_changes(self, price_data: list[float], volume_data: list[float]) -> tuple[list[float], list[float]] | None:
        """计算价格和成交量变化。

        Args:
            price_data: 价格数据序列
            volume_data: 成交量数据序列

        Returns:
            价格变化和成交量变化元组，失败时返回None
        """
        price_changes = []
        volume_changes = []

        for i in range(1, len(price_data)):
            if price_data[i-1] > 0 and volume_data[i-1] > 0:
                price_change = (price_data[i] - price_data[i-1]) / price_data[i-1]
                volume_change = (volume_data[i] - volume_data[i-1]) / volume_data[i-1]
                price_changes.append(price_change)
                volume_changes.append(volume_change)

        return (price_changes, volume_changes) if price_changes else None

    def _analyze_correlation(self, price_changes: list[float], volume_changes: list[float]) -> dict[str, Any]:
        """分析价格-成交量相关性。

        Args:
            price_changes: 价格变化序列
            volume_changes: 成交量变化序列

        Returns:
            相关性分析结果
        """
        correlation = np.corrcoef(price_changes, volume_changes)[0, 1] if len(price_changes) > 1 else 0

        # 相关性强度分类
        if abs(correlation) < 0.2:
            correlation_strength = "弱相关"
        elif abs(correlation) < 0.5:
            correlation_strength = "中等相关"
        elif abs(correlation) < 0.8:
            correlation_strength = "强相关"
        else:
            correlation_strength = "极强相关"

        correlation_direction = "正相关" if correlation > 0 else "负相关" if correlation < 0 else "无相关"

        return {
            "correlation_coefficient": correlation,
            "correlation_strength": correlation_strength,
            "correlation_direction": correlation_direction
        }

    def _identify_coordination_patterns(self, price_changes: list[float], volume_changes: list[float]) -> dict[str, Any]:
        """识别量价配合模式。

        Args:
            price_changes: 价格变化序列
            volume_changes: 成交量变化序列

        Returns:
            量价配合模式分析结果
        """
        coordination_patterns = []
        for i, (price_change, volume_change) in enumerate(zip(price_changes, volume_changes, strict=True)):
            if abs(price_change) > 0.001 and abs(volume_change) > 0.2:  # 显著变化阈值
                pattern = self._determine_coordination_pattern(price_change, volume_change)

                coordination_patterns.append({
                    "position": i + 1,
                    "pattern": pattern,
                    "price_change_percent": price_change * 100,
                    "volume_change_percent": volume_change * 100
                })

        # 统计各种模式的出现频率
        pattern_counts: dict[str, int] = {}
        for pattern_info in coordination_patterns:
            pattern = str(pattern_info["pattern"])
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        # 判断主要模式
        dominant_pattern = max(pattern_counts.items(), key=lambda x: x[1])[0] if pattern_counts else "无明显模式"

        return {
            "all_patterns": coordination_patterns,
            "pattern_counts": pattern_counts,
            "dominant_pattern": dominant_pattern,
            "total_significant_movements": len(coordination_patterns)
        }

    def _determine_coordination_pattern(self, price_change: float, volume_change: float) -> str:
        """确定量价配合模式。

        Args:
            price_change: 价格变化
            volume_change: 成交量变化

        Returns:
            量价配合模式
        """
        if price_change > 0 and volume_change > 0:
            return "价涨量增"  # 健康上涨
        elif price_change < 0 and volume_change > 0:
            return "价跌量增"  # 抛售压力
        elif price_change > 0 and volume_change < 0:
            return "价涨量缩"  # 上涨乏力
        else:
            return "价跌量缩"  # 止跌迹象

    def _analyze_pattern_distribution(self, coordination_patterns: dict[str, Any]) -> dict[str, float]:
        """分析模式分布。

        Args:
            coordination_patterns: 量价配合模式数据

        Returns:
            模式分布分析结果
        """
        pattern_counts = coordination_patterns.get("pattern_counts", {})
        total_patterns = len(coordination_patterns.get("all_patterns", []))

        if total_patterns == 0:
            return {
                "healthy_uptrend": 0,
                "selling_pressure": 0,
                "weak_uptrend": 0,
                "stabilization": 0
            }

        return {
            "healthy_uptrend": pattern_counts.get("价涨量增", 0) / total_patterns,
            "selling_pressure": pattern_counts.get("价跌量增", 0) / total_patterns,
            "weak_uptrend": pattern_counts.get("价涨量缩", 0) / total_patterns,
            "stabilization": pattern_counts.get("价跌量缩", 0) / total_patterns
        }

    def _assess_market_activity(self, minute_data_points: list[dict[str, Any]],
                              price_data: list[float], volume_data: list[float]) -> dict[str, Any]:
        """评估市场活跃度。

        Args:
            minute_data_points: 分钟数据点
            price_data: 价格数据
            volume_data: 成交量数据

        Returns:
            市场活跃度评估结果
        """
        # 计算价格水平数量（流动性分散度）
        price_levels_count = set()
        total_trades_count = 0

        for point in minute_data_points:
            price_levels = point.get("price_levels", {})
            price_levels_count.update(price_levels.keys())

            for level_data in price_levels.values():
                trades_count = level_data.get("trades_count", 0)
                total_trades_count += trades_count

        # 活跃度指标计算
        avg_volume = np.mean(volume_data) if volume_data else 0
        price_volatility = np.std(price_data) / np.mean(price_data) if len(price_data) > 1 and np.mean(price_data) > 0 else 0
        price_levels_diversity = len(price_levels_count)
        trades_per_minute = total_trades_count / len(minute_data_points) if minute_data_points else 0

        # 综合活跃度评分（0-100）
        volume_score = min(avg_volume / 10, 30)  # 成交量评分，最高30分
        volatility_score = min(price_volatility * 1000, 25)  # 波动率评分，最高25分
        diversity_score = min(price_levels_diversity / 5, 25)  # 价格多样性评分，最高25分
        trades_score = min(trades_per_minute / 2, 20)  # 交易频率评分，最高20分

        total_activity_score = volume_score + volatility_score + diversity_score + trades_score

        # 活跃度等级
        if total_activity_score >= 80:
            activity_level = "极度活跃"
        elif total_activity_score >= 60:
            activity_level = "高度活跃"
        elif total_activity_score >= 40:
            activity_level = "中等活跃"
        elif total_activity_score >= 20:
            activity_level = "低活跃"
        else:
            activity_level = "极不活跃"

        # 市场状态评估
        if avg_volume > 500 and price_volatility > 0.005:
            market_state = "高波动高成交"
        elif avg_volume > 500 and price_volatility <= 0.005:
            market_state = "低波动高成交"
        elif avg_volume <= 500 and price_volatility > 0.005:
            market_state = "高波动低成交"
        else:
            market_state = "低波动低成交"

        return {
            "activity_metrics": {
                "average_volume": avg_volume,
                "price_volatility": price_volatility,
                "price_levels_diversity": price_levels_diversity,
                "trades_per_minute": trades_per_minute
            },
            "activity_scoring": {
                "volume_score": volume_score,
                "volatility_score": volatility_score,
                "diversity_score": diversity_score,
                "trades_score": trades_score,
                "total_score": total_activity_score
            },
            "activity_assessment": {
                "activity_level": activity_level,
                "market_state": market_state,
                "data_quality": "良好" if len(minute_data_points) >= 30 else "一般" if len(minute_data_points) >= 10 else "较差"
            }
        }

    def _generate_analysis_summary(self, volatility_analysis: dict[str, Any],
                                 volume_analysis: dict[str, Any],
                                 price_volume_relation: dict[str, Any],
                                 market_activity: dict[str, Any]) -> dict[str, Any]:
        """生成综合分析摘要。

        Args:
            volatility_analysis: 波动率分析结果
            volume_analysis: 成交量分析结果
            price_volume_relation: 价格-成交量关系分析结果
            market_activity: 市场活跃度评估结果

        Returns:
            综合分析摘要
        """
        # 提取关键指标
        volatility_level = volatility_analysis.get("volatility_metrics", {}).get("volatility_level", "未知")
        volume_trend = volume_analysis.get("trend_analysis", {}).get("trend_direction", "未知")
        correlation_strength = price_volume_relation.get("correlation_analysis", {}).get("correlation_strength", "未知")
        activity_level = market_activity.get("activity_assessment", {}).get("activity_level", "未知")
        dominant_pattern = price_volume_relation.get("coordination_patterns", {}).get("dominant_pattern", "无明显模式")

        # 生成关键洞察
        key_insights = []

        # 波动率洞察
        if volatility_level in ["高", "极高"]:
            key_insights.append("市场波动率较高，价格变动剧烈")
        elif volatility_level in ["低", "极低"]:
            key_insights.append("市场波动率较低，价格相对稳定")

        # 成交量洞察
        if volume_trend == "上升":
            key_insights.append("成交量呈上升趋势，市场参与度增加")
        elif volume_trend == "下降":
            key_insights.append("成交量呈下降趋势，市场参与度减少")

        # 量价关系洞察
        if dominant_pattern == "价涨量增":
            key_insights.append("健康的价涨量增模式，上涨具有持续性")
        elif dominant_pattern == "价跌量增":
            key_insights.append("价跌量增模式，存在抛售压力")
        elif dominant_pattern == "价涨量缩":
            key_insights.append("价涨量缩模式，上涨可能缺乏动力")

        # 市场活跃度洞察
        if activity_level in ["极度活跃", "高度活跃"]:
            key_insights.append(f"市场{activity_level}，交易机会丰富")
        elif activity_level in ["低活跃", "极不活跃"]:
            key_insights.append(f"市场{activity_level}，流动性有限")

        return {
            "key_metrics": {
                "volatility_level": volatility_level,
                "volume_trend": volume_trend,
                "price_volume_correlation": correlation_strength,
                "dominant_pattern": dominant_pattern,
                "activity_level": activity_level
            },
            "key_insights": key_insights,
            "analysis_quality": {
                "data_completeness": "完整",
                "analysis_confidence": "高" if len(key_insights) >= 3 else "中",
                "recommendations": self._generate_recommendations(volatility_analysis, volume_analysis, price_volume_relation)
            }
        }

    def _generate_recommendations(self, volatility_analysis: dict[str, Any],
                                volume_analysis: dict[str, Any],
                                price_volume_relation: dict[str, Any]) -> list[str]:
        """生成交易建议。

        Args:
            volatility_analysis: 波动率分析结果
            volume_analysis: 成交量分析结果
            price_volume_relation: 价格-成交量关系分析结果

        Returns:
            交易建议列表
        """
        recommendations = []

        volatility_level = volatility_analysis.get("volatility_metrics", {}).get("volatility_level", "")
        volume_trend = volume_analysis.get("trend_analysis", {}).get("trend_direction", "")
        dominant_pattern = price_volume_relation.get("coordination_patterns", {}).get("dominant_pattern", "")

        # 基于波动率的建议
        if volatility_level in ["高", "极高"]:
            recommendations.append("高波动环境下建议控制仓位规模，注意风险管理")
        elif volatility_level in ["低", "极低"]:
            recommendations.append("低波动环境可能预示趋势即将形成，建议密切关注")

        # 基于成交量的建议
        if volume_trend == "上升":
            recommendations.append("成交量上升确认趋势强度，可考虑跟随主要趋势")
        elif volume_trend == "下降":
            recommendations.append("成交量下降可能预示趋势反转，建议谨慎操作")

        # 基于量价关系的建议
        if dominant_pattern == "价涨量增":
            recommendations.append("健康的价涨量增模式，上涨趋势有望延续")
        elif dominant_pattern == "价跌量增":
            recommendations.append("价跌量增显示抛售压力，建议避免逆势操作")
        elif dominant_pattern == "价涨量缩":
            recommendations.append("价涨量缩显示上涨乏力，建议警惕趋势反转")

        return recommendations
