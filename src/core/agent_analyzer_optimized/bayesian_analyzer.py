"""贝叶斯分析引擎 - 基于贝叶斯思维的趋势概率分析。

该模块实现贝叶斯概率更新机制，结合静态订单簿和动态交易数据：
1. 设定先验概率
2. 收集和分析证据
3. 计算似然函数
4. 贝叶斯更新后验概率
5. 输出概率化的趋势预测
"""

import logging
import math
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# 趋势类型定义
TREND_TYPES = ["震荡", "微弱看涨", "看涨", "强力看涨", "微弱看跌", "看跌", "强力看跌"]

# 先验概率（基于历史市场统计）
DEFAULT_PRIOR_PROBABILITIES = {
    "震荡": 0.30,
    "微弱看涨": 0.12,
    "看涨": 0.15,
    "强力看涨": 0.08,
    "微弱看跌": 0.12,
    "看跌": 0.15,
    "强力看跌": 0.08
}


class BayesianAnalyzer:
    """贝叶斯分析器，用于概率化趋势预测。

    该分析器基于贝叶斯定理，结合多种证据源进行概率更新，
    输出带有置信度的趋势预测结果。
    """

    def __init__(self, prior_probabilities: dict[str, float] = None):
        """初始化贝叶斯分析器。

        Args:
            prior_probabilities: 先验概率字典，如果为None则使用默认值
        """
        self.prior_probabilities = prior_probabilities or DEFAULT_PRIOR_PROBABILITIES.copy()

        # 验证先验概率
        self._validate_prior_probabilities()

        # 证据权重配置
        self.evidence_weights = {
            "static_liquidity": 0.25,      # 静态流动性证据权重
            "dynamic_volume": 0.35,        # 动态成交量证据权重
            "order_imbalance": 0.20,       # 订单不平衡证据权重
            "price_momentum": 0.20         # 价格动能证据权重
        }

        logger.info("Initialized BayesianAnalyzer with probability framework")

    def _validate_prior_probabilities(self) -> None:
        """验证先验概率的有效性。"""
        total = sum(self.prior_probabilities.values())
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Prior probabilities sum to {total:.3f}, normalizing...")
            # 归一化
            for trend in self.prior_probabilities:
                self.prior_probabilities[trend] /= total

        for trend in TREND_TYPES:
            if trend not in self.prior_probabilities:
                logger.warning(f"Missing prior probability for trend: {trend}")
                self.prior_probabilities[trend] = 0.0

    def analyze_bayesian_trend(
        self,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any],
        symbol: str = "BTCFDUSD"
    ) -> dict[str, Any]:
        """执行贝叶斯趋势分析。

        Args:
            static_data: 静态订单簿分析数据
            dynamic_data: 动态交易数据分析数据
            symbol: 交易符号

        Returns:
            贝叶斯趋势分析结果
        """
        logger.info(f"Starting Bayesian trend analysis for {symbol}")

        # 验证输入数据
        if not static_data and not dynamic_data:
            logger.warning("静态数据和动态数据都为空")
            return self._create_error_result(symbol, "No data available for analysis")

        try:
            # 第一步：提取和分析证据
            evidences = self._extract_evidences(static_data, dynamic_data)

            # 检查是否有有效证据
            if not evidences:
                logger.warning("没有提取到有效证据")
                return self._create_error_result(symbol, "No valid evidence extracted")

            # 第二步：计算每个趋势的似然函数
            likelihoods = self._calculate_likelihoods(evidences)

            # 第三步：贝叶斯更新计算后验概率
            posterior_probabilities = self._bayesian_update(likelihoods)

            # 第四步：分析结果
            analysis_result = self._analyze_bayesian_result(posterior_probabilities, evidences)

            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "analysis_type": "bayesian_trend_analysis",
                "prior_probabilities": self.prior_probabilities,
                "evidences": evidences,
                "likelihoods": likelihoods,
                "posterior_probabilities": posterior_probabilities,
                "analysis_result": analysis_result,
                "status": "success",
            }

            logger.info(
                f"Bayesian analysis completed: "
                f"trend={analysis_result['most_likely_trend']}, "
                f"confidence={analysis_result['confidence']:.3f}"
            )

            return result

        except Exception as e:
            logger.error(f"Bayesian trend analysis failed: {e}")
            return self._create_error_result(symbol, str(e))

    def _extract_evidences(
        self,
        static_data: dict[str, Any],
        dynamic_data: dict[str, Any]
    ) -> dict[str, Any]:
        """提取和分析证据。

        Args:
            static_data: 静态订单簿数据
            dynamic_data: 动态交易数据

        Returns:
            证据字典
        """
        evidences = {}

        # 静态流动性证据
        if static_data.get("status") == "success":
            evidences["static_liquidity"] = self._extract_static_liquidity_evidence(static_data)

        # 动态成交量证据
        if dynamic_data:
            evidences["dynamic_volume"] = self._extract_dynamic_volume_evidence(dynamic_data)

        # 订单不平衡证据
        if static_data.get("status") == "success":
            evidences["order_imbalance"] = self._extract_order_imbalance_evidence(static_data)

        # 价格动能证据
        if dynamic_data:
            evidences["price_momentum"] = self._extract_price_momentum_evidence(dynamic_data)

        return evidences

    def _extract_static_liquidity_evidence(self, static_data: dict[str, Any]) -> dict[str, Any]:
        """提取静态流动性证据。"""
        liquidity_analysis = static_data.get("liquidity_analysis", {})
        key_levels = static_data.get("key_levels", {})

        bid_volume = liquidity_analysis.get("bid_side", {}).get("total_volume", 0)
        ask_volume = liquidity_analysis.get("ask_side", {}).get("total_volume", 0)
        total_liquidity = liquidity_analysis.get("total_liquidity", 0)

        bid_concentration = liquidity_analysis.get("bid_side", {}).get("concentration", 0)
        ask_concentration = liquidity_analysis.get("ask_side", {}).get("concentration", 0)

        # 流动性墙强度
        strongest_support_volume = key_levels.get("strongest_support", {}).get("volume", 0)
        strongest_resistance_volume = key_levels.get("strongest_resistance", {}).get("volume", 0)

        # 安全的bid_ask_ratio计算
        if ask_volume > 0:
            bid_ask_ratio = bid_volume / ask_volume
        elif bid_volume > 0:
            bid_ask_ratio = float('inf')
        else:
            bid_ask_ratio = 1.0  # 无数据时默认为1:1

        return {
            "bid_volume": bid_volume,
            "ask_volume": ask_volume,
            "total_liquidity": total_liquidity,
            "bid_ask_ratio": bid_ask_ratio,
            "bid_concentration": bid_concentration,
            "ask_concentration": ask_concentration,
            "strongest_support_volume": strongest_support_volume,
            "strongest_resistance_volume": strongest_resistance_volume,
            "liquidity_strength": "high" if total_liquidity > 1000 else "medium" if total_liquidity > 100 else "low"
        }

    def _extract_dynamic_volume_evidence(self, dynamic_data: dict[str, Any]) -> dict[str, Any]:
        """提取动态成交量证据。"""
        minute_data_points = dynamic_data.get("minute_data_points", [])
        data_points_count = dynamic_data.get("data_points_count", len(minute_data_points))

        if not minute_data_points:
            return {
                "total_volume": 0,
                "volume_trend": "unknown",
                "volume_volatility": 0,
                "recent_activity": "low",
                "data_points_count": data_points_count
            }

        # 计算成交量趋势
        volumes = []
        for point in minute_data_points[-20:]:  # 最近20个数据点
            price_levels = point.get("price_levels", {})
            point_volume = sum(
                float(level.get("total_volume", 0))
                for level in price_levels.values()
            )
            volumes.append(point_volume)

        total_volume = sum(volumes)
        avg_volume = sum(volumes) / len(volumes) if volumes else 0

        # 成交量趋势分析
        if len(volumes) >= 10:
            recent_avg = sum(volumes[-5:]) / 5
            earlier_avg = sum(volumes[-10:-5]) / 5
            volume_trend = "increasing" if recent_avg > earlier_avg * 1.2 else "decreasing" if recent_avg < earlier_avg * 0.8 else "stable"
        else:
            volume_trend = "unknown"

        # 成交量波动性
        if len(volumes) > 1:
            variance = sum((v - avg_volume) ** 2 for v in volumes) / len(volumes)
            volume_volatility = math.sqrt(variance) / avg_volume if avg_volume > 0 else 0
        else:
            volume_volatility = 0

        # 近期活跃度
        recent_activity = "high" if total_volume > 500 else "medium" if total_volume > 100 else "low"

        return {
            "total_volume": total_volume,
            "volume_trend": volume_trend,
            "volume_volatility": volume_volatility,
            "recent_activity": recent_activity,
            "data_points_count": data_points_count
        }

    def _extract_order_imbalance_evidence(self, static_data: dict[str, Any]) -> dict[str, Any]:
        """提取订单不平衡证据。"""
        imbalance_metrics = static_data.get("imbalance_metrics", {})

        return {
            "bid_ask_ratio": imbalance_metrics.get("bid_ask_ratio", 1.0),
            "bid_percentage": imbalance_metrics.get("bid_percentage", 0.5),
            "ask_percentage": imbalance_metrics.get("ask_percentage", 0.5),
            "imbalance_strength": imbalance_metrics.get("imbalance_strength", 0.0),
            "direction": imbalance_metrics.get("direction", "neutral")
        }

    def _extract_price_momentum_evidence(self, dynamic_data: dict[str, Any]) -> dict[str, Any]:
        """提取价格动能证据。"""
        minute_data_points = dynamic_data.get("minute_data_points", [])

        if not minute_data_points:
            return {
                "momentum_direction": "neutral",
                "momentum_strength": 0.0,
                "price_trend": "unknown"
            }

        # 计算价格动能
        prices = []
        for point in minute_data_points[-20:]:  # 最近20个数据点
            price_levels = point.get("price_levels", {})
            if price_levels:
                # 使用成交量加权平均价格
                total_volume = 0
                weighted_price = 0
                for price_str, level_data in price_levels.items():
                    volume = float(level_data.get("total_volume", 0))
                    if volume > 0:
                        price = float(price_str)
                        weighted_price += price * volume
                        total_volume += volume

                if total_volume > 0:
                    vwap = weighted_price / total_volume
                    prices.append(vwap)

        if len(prices) < 3:
            return {
                "momentum_direction": "neutral",
                "momentum_strength": 0.0,
                "price_trend": "unknown"
            }

        # 计算价格趋势
        recent_prices = prices[-5:]
        earlier_prices = prices[-10:-5] if len(prices) >= 10 else prices[:-5]

        recent_avg = sum(recent_prices) / len(recent_prices)
        earlier_avg = sum(earlier_prices) / len(earlier_prices)

        price_change = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0

        # 动能方向和强度
        if price_change > 0.002:  # 0.2%以上
            momentum_direction = "bullish"
            momentum_strength = min(abs(price_change) * 100, 1.0)
        elif price_change < -0.002:
            momentum_direction = "bearish"
            momentum_strength = min(abs(price_change) * 100, 1.0)
        else:
            momentum_direction = "neutral"
            momentum_strength = 0.0

        price_trend = "upward" if price_change > 0.001 else "downward" if price_change < -0.001 else "sideways"

        return {
            "momentum_direction": momentum_direction,
            "momentum_strength": momentum_strength,
            "price_trend": price_trend,
            "price_change": price_change
        }

    def _calculate_likelihoods(self, evidences: dict[str, Any]) -> dict[str, dict[str, float]]:
        """计算每个趋势的似然函数。

        Args:
            evidences: 证据字典

        Returns:
            每个趋势类型的似然概率字典
        """
        likelihoods = {}

        for trend in TREND_TYPES:
            trend_likelihoods = {}

            # 静态流动性似然
            if "static_liquidity" in evidences:
                trend_likelihoods["static_liquidity"] = self._calculate_static_liquidity_likelihood(
                    evidences["static_liquidity"], trend
                )

            # 动态成交量似然
            if "dynamic_volume" in evidences:
                trend_likelihoods["dynamic_volume"] = self._calculate_dynamic_volume_likelihood(
                    evidences["dynamic_volume"], trend
                )

            # 订单不平衡似然
            if "order_imbalance" in evidences:
                trend_likelihoods["order_imbalance"] = self._calculate_order_imbalance_likelihood(
                    evidences["order_imbalance"], trend
                )

            # 价格动能似然
            if "price_momentum" in evidences:
                trend_likelihoods["price_momentum"] = self._calculate_price_momentum_likelihood(
                    evidences["price_momentum"], trend
                )

            likelihoods[trend] = trend_likelihoods

        return likelihoods

    def _calculate_static_liquidity_likelihood(self, evidence: dict[str, Any], trend: str) -> float:
        """计算静态流动性似然。"""
        bid_ask_ratio = evidence.get("bid_ask_ratio", 1.0)
        liquidity_strength = evidence.get("liquidity_strength", "medium")

        # 基础概率
        base_likelihood = 0.5

        # 根据趋势调整
        if "看涨" in trend:
            if bid_ask_ratio > 1.2:
                base_likelihood = 0.8
            elif bid_ask_ratio > 1.0:
                base_likelihood = 0.6
            else:
                base_likelihood = 0.3
        elif "看跌" in trend:
            if bid_ask_ratio < 0.8:
                base_likelihood = 0.8
            elif bid_ask_ratio < 1.0:
                base_likelihood = 0.6
            else:
                base_likelihood = 0.3
        else:  # 震荡
            if 0.8 <= bid_ask_ratio <= 1.2:
                base_likelihood = 0.7
            else:
                base_likelihood = 0.4

        # 流动性强度调整
        if liquidity_strength == "high":
            base_likelihood *= 1.2
        elif liquidity_strength == "low":
            base_likelihood *= 0.8

        return min(base_likelihood, 1.0)

    def _calculate_dynamic_volume_likelihood(self, evidence: dict[str, Any], trend: str) -> float:
        """计算动态成交量似然。"""
        volume_trend = evidence.get("volume_trend", "unknown")
        recent_activity = evidence.get("recent_activity", "low")
        volume_volatility = evidence.get("volume_volatility", 0)

        base_likelihood = 0.5

        # 成交量趋势匹配
        if "看涨" in trend and volume_trend == "increasing":
            base_likelihood = 0.8
        elif "看跌" in trend and volume_trend == "decreasing":
            base_likelihood = 0.8
        elif trend == "震荡" and volume_trend == "stable":
            base_likelihood = 0.7
        else:
            base_likelihood = 0.4

        # 活跃度调整
        if recent_activity == "high":
            base_likelihood *= 1.1
        elif recent_activity == "low":
            base_likelihood *= 0.9

        # 波动率调整
        if trend in ["强力看涨", "强力看跌"] and volume_volatility > 0.5:
            base_likelihood *= 1.2
        elif trend == "震荡" and volume_volatility < 0.3:
            base_likelihood *= 1.1

        return min(base_likelihood, 1.0)

    def _calculate_order_imbalance_likelihood(self, evidence: dict[str, Any], trend: str) -> float:
        """计算订单不平衡似然。"""
        direction = evidence.get("direction", "neutral")
        imbalance_strength = evidence.get("imbalance_strength", 0.0)

        base_likelihood = 0.5

        # 方向匹配
        if "看涨" in trend and direction == "bullish":
            base_likelihood = 0.7 + imbalance_strength * 0.3
        elif "看跌" in trend and direction == "bearish":
            base_likelihood = 0.7 + imbalance_strength * 0.3
        elif trend == "震荡" and direction == "neutral":
            base_likelihood = 0.8
        else:
            base_likelihood = 0.3

        return min(base_likelihood, 1.0)

    def _calculate_price_momentum_likelihood(self, evidence: dict[str, Any], trend: str) -> float:
        """计算价格动能似然。"""
        momentum_direction = evidence.get("momentum_direction", "neutral")
        momentum_strength = evidence.get("momentum_strength", 0.0)

        base_likelihood = 0.5

        # 动能方向匹配
        if "看涨" in trend and momentum_direction == "bullish":
            base_likelihood = 0.6 + momentum_strength * 0.4
        elif "看跌" in trend and momentum_direction == "bearish":
            base_likelihood = 0.6 + momentum_strength * 0.4
        elif trend == "震荡" and momentum_direction == "neutral":
            base_likelihood = 0.7
        else:
            base_likelihood = 0.3

        return min(base_likelihood, 1.0)

    def _bayesian_update(self, likelihoods: dict[str, dict[str, float]]) -> dict[str, float]:
        """执行贝叶斯更新计算后验概率。

        Args:
            likelihoods: 似然函数字典

        Returns:
            后验概率字典
        """
        posterior_probabilities = {}

        for trend in TREND_TYPES:
            # 计算该趋势的似然（考虑证据权重）
            trend_likelihood = 1.0
            trend_evidences = likelihoods.get(trend, {})

            for evidence_type, likelihood in trend_evidences.items():
                weight = self.evidence_weights.get(evidence_type, 0.25)
                # 加权几何平均
                trend_likelihood *= (likelihood ** weight)

            # 贝叶斯更新: P(trend|evidence) ∝ P(evidence|trend) * P(trend)
            prior = self.prior_probabilities.get(trend, 0.0)
            posterior = trend_likelihood * prior
            posterior_probabilities[trend] = posterior

        # 归一化后验概率
        total_posterior = sum(posterior_probabilities.values())
        if total_posterior > 0:
            for trend in posterior_probabilities:
                posterior_probabilities[trend] /= total_posterior
        else:
            # 如果计算失败，返回先验概率
            posterior_probabilities = self.prior_probabilities.copy()

        return posterior_probabilities

    def _analyze_bayesian_result(
        self,
        posterior_probabilities: dict[str, float],
        evidences: dict[str, Any]
    ) -> dict[str, Any]:
        """分析贝叶斯结果。

        Args:
            posterior_probabilities: 后验概率字典
            evidences: 证据字典

        Returns:
            分析结果字典
        """
        # 找出最可能的趋势
        most_likely_trend = max(posterior_probabilities.items(), key=lambda x: x[1])
        trend, probability = most_likely_trend

        # 计算置信度
        confidence = probability
        if confidence > 0.7:
            confidence_level = "high"
        elif confidence > 0.5:
            confidence_level = "medium"
        else:
            confidence_level = "low"

        # 计算概率分布的离散程度（不确定性）
        entropy = -sum(p * math.log(p + 1e-10) for p in posterior_probabilities.values() if p > 0)
        max_entropy = -len(TREND_TYPES) * (1/len(TREND_TYPES)) * math.log(1/len(TREND_TYPES))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0

        # 生成分析原因
        reason = self._generate_analysis_reason(posterior_probabilities, evidences)

        return {
            "most_likely_trend": trend,
            "probability": probability,
            "confidence": confidence,
            "confidence_level": confidence_level,
            "uncertainty": uncertainty,
            "probability_distribution": posterior_probabilities,
            "analysis_reason": reason,
            "evidence_summary": self._summarize_evidences(evidences)
        }

    def _generate_analysis_reason(
        self,
        posterior_probabilities: dict[str, float],
        evidences: dict[str, Any]
    ) -> str:
        """生成分析原因。"""
        most_likely = max(posterior_probabilities.items(), key=lambda x: x[1])
        trend, probability = most_likely

        reasons = []

        # 基于证据生成原因
        if "order_imbalance" in evidences:
            imbalance = evidences["order_imbalance"]
            if imbalance["direction"] == "bullish" and "看涨" in trend:
                reasons.append(f"订单不平衡偏向买方 ({imbalance['bid_percentage']:.1%} vs {imbalance['ask_percentage']:.1%})")
            elif imbalance["direction"] == "bearish" and "看跌" in trend:
                reasons.append(f"订单不平衡偏向卖方 ({imbalance['ask_percentage']:.1%} vs {imbalance['bid_percentage']:.1%})")

        if "dynamic_volume" in evidences:
            volume = evidences["dynamic_volume"]
            if volume["volume_trend"] == "increasing" and "看涨" in trend:
                reasons.append("成交量呈现增长趋势")
            elif volume["volume_trend"] == "decreasing" and "看跌" in trend:
                reasons.append("成交量呈现下降趋势")

        if "price_momentum" in evidences:
            momentum = evidences["price_momentum"]
            if momentum["momentum_direction"] == "bullish" and "看涨" in trend:
                reasons.append(f"价格动能偏向上涨 ({momentum['price_change']:.2%})")
            elif momentum["momentum_direction"] == "bearish" and "看跌" in trend:
                reasons.append(f"价格动能偏向下跌 ({momentum['price_change']:.2%})")

        if not reasons:
            reasons.append("基于多个证据的综合分析")

        return f"贝叶斯分析显示趋势为'{trend}'（概率{probability:.1%}），主要依据：{'; '.join(reasons)}"

    def _summarize_evidences(self, evidences: dict[str, Any]) -> dict[str, str]:
        """总结证据。"""
        summary = {}

        if "static_liquidity" in evidences:
            liquidity = evidences["static_liquidity"]
            summary["static_liquidity"] = f"流动性强度{liquidity['liquidity_strength']}，买卖比{liquidity['bid_ask_ratio']:.2f}"

        if "dynamic_volume" in evidences:
            volume = evidences["dynamic_volume"]
            summary["dynamic_volume"] = f"成交量趋势{volume['volume_trend']}，活跃度{volume['recent_activity']}"

        if "order_imbalance" in evidences:
            imbalance = evidences["order_imbalance"]
            summary["order_imbalance"] = f"订单不平衡方向{imbalance['direction']}，强度{imbalance['imbalance_strength']:.2f}"

        if "price_momentum" in evidences:
            momentum = evidences["price_momentum"]
            summary["price_momentum"] = f"价格动能{momentum['momentum_direction']}，强度{momentum['momentum_strength']:.2f}"

        return summary

    def _create_error_result(self, symbol: str, error_message: str) -> dict[str, Any]:
        """创建错误结果。"""
        return {
            "symbol": symbol,
            "analysis_type": "bayesian_trend_analysis",
            "status": "error",
            "error": error_message,
            "prior_probabilities": {},
            "evidences": {},
            "likelihoods": {},
            "posterior_probabilities": {},
            "analysis_result": {}
        }
