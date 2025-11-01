"""贝叶斯响应格式化器 - 专门处理贝叶斯分析结果格式化。

该模块提供贝叶斯分析结果的标准格式化输出：
1. 贝叶斯趋势结果的JSON格式化
2. 概率分布的清晰展示
3. 置信度和不确定性的可视化
4. 证据权重和推理链条展示
"""

import json
import logging
from datetime import datetime
from typing import Any

# 导入BayesianTrendResult以避免循环导入
if False:  # 防止运行时导入
    pass

logger = logging.getLogger(__name__)


class BayesianResponseFormatter:
    """贝叶斯响应格式化器，专门处理概率化分析结果。

    该格式化器提供：
    1. 贝叶斯分析结果的标准JSON输出
    2. 概率分布的美观格式化
    3. 证据分析和推理链条展示
    4. 置信度和不确定性的量化显示
    """

    def __init__(self, include_metadata: bool = True, pretty_print: bool = False):
        """初始化贝叶斯响应格式化器。

        Args:
            include_metadata: 是否包含元数据
            pretty_print: 是否美化JSON输出
        """
        self.include_metadata = include_metadata
        self.pretty_print = pretty_print

        logger.info(
            f"Initialized BayesianResponseFormatter: include_metadata={include_metadata}, "
            f"pretty_print={pretty_print}"
        )

    def format_bayesian_response(
        self,
        bayesian_result: dict[str, Any],
        static_data: dict[str, Any] | None = None,
        dynamic_data: dict[str, Any] | None = None,
        symbol: str = "BTCFDUSD"
    ) -> str:
        """格式化贝叶斯分析响应。

        Args:
            bayesian_result: 贝叶斯趋势分析结果
            static_data: 静态订单簿数据（可选）
            dynamic_data: 动态交易数据（可选）
            symbol: 交易符号

        Returns:
            格式化的JSON响应字符串
        """
        try:
            # 构建标准响应结构
            response_data = {
                "timestamp": bayesian_result.timestamp.isoformat(),
                "symbol": symbol,
                "analysis_type": "bayesian_trend_analysis",
                "trend_analysis": {
                    "most_likely_trend": bayesian_result.most_likely_trend,
                    "confidence": bayesian_result.confidence,
                    "confidence_level": self._get_confidence_level(bayesian_result.confidence),
                    "uncertainty": bayesian_result.uncertainty,
                    "risk_level": self._get_risk_level(bayesian_result.uncertainty)
                },
                "probability_distribution": self._format_probability_distribution(
                    bayesian_result.posterior_probabilities
                ),
                "bayesian_analysis": {
                    "analysis_reason": bayesian_result.analysis_reason,
                    "evidence_summary": bayesian_result.evidence_summary,
                    "key_insights": self._extract_key_insights(bayesian_result),
                    "probability_drivers": self._identify_probability_drivers(
                        bayesian_result.posterior_probabilities
                    )
                }
            }

            # 添加证据权重分析
            if static_data or dynamic_data:
                response_data["evidence_analysis"] = self._format_evidence_analysis(
                    static_data, dynamic_data
                )

            # 添加元数据（如果启用）
            if self.include_metadata:
                response_data["metadata"] = {
                    "analysis_method": "bayesian_inference",
                    "data_sources": self._get_data_sources(static_data, dynamic_data),
                    "analysis_timestamp": datetime.now().isoformat(),
                    "model_type": "bayesian_deepseek",
                    **bayesian_result.bayesian_metadata
                }

            # 添加格式化建议
            response_data["recommendations"] = self._generate_bayesian_recommendations(
                bayesian_result
            )

            # 转换为JSON
            if self.pretty_print:
                json_response = json.dumps(response_data, indent=2, ensure_ascii=False)
            else:
                json_response = json.dumps(response_data, ensure_ascii=False)

            logger.info(
                f"Bayesian response formatted: trend={bayesian_result.most_likely_trend}, "
                f"confidence={bayesian_result.confidence:.3f}"
            )

            return json_response

        except Exception as e:
            logger.error(f"Bayesian response formatting failed: {e}")
            return self._format_error_response(str(e), symbol)

    def _format_probability_distribution(self, probabilities: dict[str, float]) -> dict[str, Any]:
        """格式化概率分布。"""
        # 排序概率分布
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        # 计算统计信息
        top_three = sorted_probs[:3]
        entropy = self._calculate_entropy(probabilities)

        return {
            "full_distribution": dict(sorted_probs),
            "top_three_trends": [
                {"trend": trend, "probability": prob}
                for trend, prob in top_three
            ],
            "entropy": entropy,
            "distribution_type": self._classify_distribution_type(probabilities),
            "probability_spread": self._calculate_probability_spread(probabilities)
        }

    def _calculate_entropy(self, probabilities: dict[str, float]) -> float:
        """计算概率分布的熵。"""
        import math
        entropy = 0.0
        for p in probabilities.values():
            if p > 0:
                entropy -= p * math.log(p)
        return entropy

    def _classify_distribution_type(self, probabilities: dict[str, float]) -> str:
        """分类概率分布类型。"""
        max_prob = max(probabilities.values())
        second_max = sorted(probabilities.values())[-2]

        if max_prob > 0.7:
            return "highly_concentrated"
        elif max_prob > 0.5:
            return "moderately_concentrated"
        elif max_prob - second_max < 0.1:
            return "highly_uncertain"
        else:
            return "moderately_uncertain"

    def _calculate_probability_spread(self, probabilities: dict[str, float]) -> float:
        """计算概率分布的离散程度。"""
        probs = list(probabilities.values())
        import statistics
        return statistics.stdev(probs) if len(probs) > 1 else 0.0

    def _get_confidence_level(self, confidence: float) -> str:
        """获取置信度等级。"""
        if confidence > 0.8:
            return "very_high"
        elif confidence > 0.6:
            return "high"
        elif confidence > 0.4:
            return "medium"
        elif confidence > 0.2:
            return "low"
        else:
            return "very_low"

    def _get_risk_level(self, uncertainty: float) -> str:
        """获取风险等级。"""
        if uncertainty < 0.2:
            return "low_risk"
        elif uncertainty < 0.4:
            return "moderate_risk"
        elif uncertainty < 0.6:
            return "high_risk"
        else:
            return "very_high_risk"

    def _extract_key_insights(self, bayesian_result: dict[str, Any]) -> dict[str, Any]:
        """提取关键洞察。"""
        insights = {
            "primary_driver": "unknown",
            "evidence_consistency": "unknown",
            "contradictory_signals": [],
            "strength_factors": [],
            "weakness_factors": []
        }

        # 基于分析原因提取洞察
        reason = bayesian_result.analysis_reason.lower()

        if "订单不平衡" in reason or "买方" in reason or "卖方" in reason:
            insights["primary_driver"] = "order_imbalance"
            insights["strength_factors"].append("流动性结构明确")

        if "成交量" in reason and ("增长" in reason or "增加" in reason):
            insights["strength_factors"].append("成交量趋势确认")

        if "动能" in reason or "价格" in reason:
            insights["primary_driver"] = "price_momentum"

        # 检查证据一致性
        evidence_summary = bayesian_result.evidence_summary
        if len(evidence_summary) > 1:
            insights["evidence_consistency"] = "multiple_evidence_support"
        else:
            insights["evidence_consistency"] = "limited_evidence"

        # 置信度分析
        if bayesian_result.confidence > 0.7:
            insights["strength_factors"].append("高置信度预测")
        elif bayesian_result.confidence < 0.4:
            insights["weakness_factors"].append("低置信度预测")

        # 不确定性分析
        if bayesian_result.uncertainty > 0.6:
            insights["weakness_factors"].append("高不确定性")
            insights["contradictory_signals"].append("证据存在矛盾")

        return insights

    def _identify_probability_drivers(self, probabilities: dict[str, float]) -> list[dict[str, Any]]:
        """识别概率驱动因素。"""
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        drivers = []
        for i, (trend, prob) in enumerate(sorted_probs[:3]):
            driver_type = self._classify_driver_type(trend, prob)
            drivers.append({
                "trend": trend,
                "probability": prob,
                "driver_type": driver_type,
                "influence_strength": self._calculate_influence_strength(prob, i)
            })

        return drivers

    def _classify_driver_type(self, trend: str, probability: float) -> str:
        """分类驱动因素类型。"""
        if probability > 0.6:
            return "dominant_driver"
        elif probability > 0.3:
            return "significant_driver"
        else:
            return "minor_driver"

    def _calculate_influence_strength(self, probability: float, rank: int) -> str:
        """计算影响力强度。"""
        if rank == 0 and probability > 0.5:
            return "very_strong"
        elif rank == 0:
            return "strong"
        elif rank == 1 and probability > 0.3:
            return "moderate"
        else:
            return "weak"

    def _format_evidence_analysis(
        self,
        static_data: dict[str, Any] | None,
        dynamic_data: dict[str, Any] | None
    ) -> dict[str, Any]:
        """格式化证据分析。"""
        evidence_analysis = {
            "available_evidence": [],
            "evidence_quality": {},
            "data_completeness": "unknown"
        }

        if static_data and static_data.get("status") == "success":
            evidence_analysis["available_evidence"].append("static_order_book")
            liquidity_quality = self._assess_static_data_quality(static_data)
            evidence_analysis["evidence_quality"]["static_liquidity"] = liquidity_quality

        if dynamic_data and dynamic_data.get("data_points_count", 0) > 0:
            evidence_analysis["available_evidence"].append("dynamic_volume_profile")
            volume_quality = self._assess_dynamic_data_quality(dynamic_data)
            evidence_analysis["evidence_quality"]["dynamic_volume"] = volume_quality

        # 评估数据完整性
        evidence_count = len(evidence_analysis["available_evidence"])
        if evidence_count == 2:
            evidence_analysis["data_completeness"] = "comprehensive"
        elif evidence_count == 1:
            evidence_analysis["data_completeness"] = "partial"
        else:
            evidence_analysis["data_completeness"] = "insufficient"

        return evidence_analysis

    def _assess_static_data_quality(self, static_data: dict[str, Any]) -> str:
        """评估静态数据质量。"""
        liquidity = static_data.get("liquidity_analysis", {})
        total_liquidity = liquidity.get("total_liquidity", 0)

        if total_liquidity > 1000:
            return "high_quality"
        elif total_liquidity > 100:
            return "medium_quality"
        else:
            return "low_quality"

    def _assess_dynamic_data_quality(self, dynamic_data: dict[str, Any]) -> str:
        """评估动态数据质量。"""
        data_points = dynamic_data.get("data_points_count", 0)
        minute_data = dynamic_data.get("minute_data_points", [])

        if data_points > 100 and len(minute_data) > 50:
            return "high_quality"
        elif data_points > 20 and len(minute_data) > 10:
            return "medium_quality"
        else:
            return "low_quality"

    def _get_data_sources(
        self,
        static_data: dict[str, Any] | None,
        dynamic_data: dict[str, Any] | None
    ) -> list[str]:
        """获取数据源列表。"""
        sources = []

        if static_data:
            sources.append("depth_snapshot_5000")

        if dynamic_data:
            sources.append("trades_window")

        return sources

    def _generate_bayesian_recommendations(self, bayesian_result: dict[str, Any]) -> dict[str, Any]:
        """生成贝叶斯分析建议。"""
        recommendations = {
            "trading_suggestions": [],
            "risk_management": [],
            "monitoring_points": [],
            "confidence_based_actions": []
        }

        trend = bayesian_result.most_likely_trend
        confidence = bayesian_result.confidence
        uncertainty = bayesian_result.uncertainty

        # 基于趋势的建议
        if "看涨" in trend:
            if confidence > 0.6:
                recommendations["trading_suggestions"].append("考虑逢低买入策略")
            recommendations["monitoring_points"].append("关注支撑位守住情况")
        elif "看跌" in trend:
            if confidence > 0.6:
                recommendations["trading_suggestions"].append("考虑高位减仓策略")
            recommendations["monitoring_points"].append("关注阻力位突破情况")
        else:
            recommendations["trading_suggestions"].append("建议区间操作策略")
            recommendations["monitoring_points"].append("关注突破方向选择")

        # 基于置信度的建议
        if confidence > 0.7:
            recommendations["confidence_based_actions"].append("可以适当增加仓位")
        elif confidence < 0.4:
            recommendations["confidence_based_actions"].append("建议降低仓位规模")
            recommendations["risk_management"].append("提高止损保护")

        # 基于不确定性的建议
        if uncertainty > 0.6:
            recommendations["risk_management"].append("建议分批建仓")
            recommendations["risk_management"].append("密切关注市场变化")
            recommendations["monitoring_points"].append("增加监控频率")

        return recommendations

    def validate_bayesian_response(self, response: str) -> bool:
        """验证贝叶斯响应格式。

        Args:
            response: JSON响应字符串

        Returns:
            是否为有效的贝叶斯响应格式
        """
        try:
            data = json.loads(response)

            # 检查必需字段
            required_fields = [
                "timestamp", "symbol", "analysis_type",
                "trend_analysis", "probability_distribution"
            ]

            for field in required_fields:
                if field not in data:
                    logger.error(f"Missing required field in Bayesian response: {field}")
                    return False

            # 检查趋势分析字段
            trend_analysis = data["trend_analysis"]
            required_trend_fields = ["most_likely_trend", "confidence", "uncertainty"]

            for field in required_trend_fields:
                if field not in trend_analysis:
                    logger.error(f"Missing trend analysis field: {field}")
                    return False

            # 检查概率分布字段
            prob_dist = data["probability_distribution"]
            if "full_distribution" not in prob_dist:
                logger.error("Missing probability distribution field")
                return False

            # 验证概率分布总和
            full_dist = prob_dist["full_distribution"]
            total_prob = sum(full_dist.values())
            if abs(total_prob - 1.0) > 0.1:
                logger.warning(f"Probability distribution sum not equal to 1: {total_prob}")

            return True

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in Bayesian response: {e}")
            return False
        except Exception as e:
            logger.error(f"Bayesian response validation failed: {e}")
            return False

    def _format_error_response(self, error_message: str, symbol: str) -> str:
        """格式化错误响应。

        Args:
            error_message: 错误消息
            symbol: 交易符号

        Returns:
            错误响应JSON字符串
        """
        error_response = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "analysis_type": "bayesian_trend_analysis",
            "status": "error",
            "error": error_message,
            "trend_analysis": {
                "most_likely_trend": "unknown",
                "confidence": 0.0,
                "uncertainty": 1.0
            },
            "probability_distribution": {
                "full_distribution": {},
                "error": "Analysis failed"
            }
        }

        return json.dumps(error_response, ensure_ascii=False)
