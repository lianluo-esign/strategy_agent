"""AI client for DeepSeek integration."""

import json
import logging
from datetime import datetime
from decimal import Decimal

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ..core.models import MarketAnalysisResult, TradingRecommendation

logger = logging.getLogger(__name__)


class DeepSeekClient:
    """Client for DeepSeek AI API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ):
        """Initialize DeepSeek client."""
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.client = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def analyze_market_data(
        self,
        analysis_result: MarketAnalysisResult,
        symbol: str = "BTCFDUSD"
    ) -> TradingRecommendation | None:
        """Analyze market data using DeepSeek AI."""
        try:
            # Prepare system prompt
            system_prompt = self._get_system_prompt()

            # Prepare user prompt with market data
            user_prompt = self._format_market_data_prompt(analysis_result, symbol)

            # Prepare tools for function calling
            tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "depth_snapshot_analysis",
                        "description": "Analyze depth snapshot data to identify static support/resistance levels",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "support_levels": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "price": {"type": "number"},
                                            "strength": {"type": "number"},
                                            "volume": {"type": "number"}
                                        }
                                    }
                                },
                                "resistance_levels": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "price": {"type": "number"},
                                            "strength": {"type": "number"},
                                            "volume": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "orderflow_analysis",
                        "description": "Analyze order flow data to confirm dynamic support/resistance and identify POCs",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "poc_levels": {
                                    "type": "array",
                                    "items": {"type": "number"}
                                },
                                "confirmed_levels": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "price": {"type": "number"},
                                            "type": {"type": "string"},
                                            "confirmation_strength": {"type": "number"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            ]

            # Make API request
            request_data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "tools": tools,
                "tool_choice": "auto",
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }

            response = await self.client.post("/chat/completions", json=request_data)
            response.raise_for_status()

            result = response.json()

            # Process response
            recommendation = self._process_ai_response(result, symbol)

            logger.info(f"AI analysis completed for {symbol}: {recommendation.action}")
            return recommendation

        except Exception as e:
            logger.error(f"DeepSeek AI analysis failed: {e}")
            return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for market analysis."""
        return """You are an expert cryptocurrency market analyst specializing in BTC-FDUSD spot trading on Binance.

Your task is to analyze market data and provide liquidity deployment recommendations using a sophisticated market-making approach.

ANALYSIS FRAMEWORK:
1. First, analyze depth snapshot data to identify static support/resistance levels
2. Then, analyze order flow data to confirm dynamic levels and find resonance zones
3. Finally, provide specific market-making recommendations based on the convergence of multiple signals

MAKING LOGIC:
- Place orders slightly ahead of large walls (e.g., if there's a wall at $50,000, place orders at $49,999.99)
- Prioritize zones with high volume concentration and repeated confirmation
- Consider liquidity vacuum zones as high-risk areas to avoid
- Focus on resonance zones where static and dynamic analysis align

RECOMMENDATION FORMAT:
Provide specific actionable recommendations including:
- Action: BUY/SELL/HOLD
- Price range: Optimal deployment range
- Confidence level: 0.0-1.0
- Reasoning: Detailed analysis of supporting factors
- Risk assessment: LOW/MEDIUM/HIGH

Always prioritize capital preservation and optimal execution timing."""

    def _format_market_data_prompt(self, analysis_result: MarketAnalysisResult, symbol: str) -> str:
        """Format market analysis data for AI prompt."""
        prompt = f"""Please analyze the following market data for {symbol} and provide market-making recommendations.

MARKET DATA:
Timestamp: {analysis_result.timestamp}
Symbol: {analysis_result.symbol}

STATIC SUPPORT LEVELS (from depth snapshot):
{self._format_levels(analysis_result.support_levels)}

STATIC RESISTANCE LEVELS (from depth snapshot):
{self._format_levels(analysis_result.resistance_levels)}

DYNAMIC CONFIRMATION (from order flow):
Point of Control Levels: {[float(poc) for poc in analysis_result.poc_levels]}
Liquidity Vacuum Zones: {[float(zone) for zone in analysis_result.liquidity_vacuum_zones]}

RESONANCE ZONES (high-probability areas):
{[float(zone) for zone in analysis_result.resonance_zones]}

ANALYSIS TASKS:
1. Use the depth_snapshot_analysis tool to analyze static support/resistance levels
2. Use the orderflow_analysis tool to analyze dynamic confirmation and POCs
3. Based on both tool results, provide specific market-making recommendations

Focus on identifying optimal liquidity deployment zones where multiple signals converge."""

        return prompt

    def _format_levels(self, levels: list) -> str:
        """Format support/resistance levels for prompt."""
        if not levels:
            return "None identified"

        formatted = []
        for level in levels[:5]:  # Limit to top 5 levels
            formatted.append(
                f"- Price: ${float(level.price):,.2f}, "
                f"Strength: {level.strength:.2f}, "
                f"Volume: {float(level.volume_at_level):.2f}, "
                f"Confirmations: {level.confirmation_count}"
            )
        return "\n".join(formatted)

    def _process_ai_response(self, response: dict, symbol: str) -> TradingRecommendation:
        """Process AI response and extract trading recommendation."""
        try:
            message = response["choices"][0]["message"]

            # Check if tools were called
            if "tool_calls" in message and message["tool_calls"]:
                # Process tool calls
                self._process_tool_calls(message["tool_calls"])

                # Get final recommendation from content
                if message.get("content"):
                    return self._parse_recommendation_from_content(
                        message["content"], symbol
                    )

            # Fallback: parse recommendation from content directly
            if message.get("content"):
                return self._parse_recommendation_from_content(
                    message["content"], symbol
                )

            # Default fallback
            return TradingRecommendation(
                timestamp=datetime.now(),
                symbol=symbol,
                action="HOLD",
                price_range=(Decimal("0"), Decimal("0")),
                confidence=0.0,
                reasoning="Unable to parse AI recommendation",
                risk_level="HIGH"
            )

        except Exception as e:
            logger.error(f"Failed to process AI response: {e}")
            return TradingRecommendation(
                timestamp=datetime.now(),
                symbol=symbol,
                action="HOLD",
                price_range=(Decimal("0"), Decimal("0")),
                confidence=0.0,
                reasoning="Error processing AI response",
                risk_level="HIGH"
            )

    def _process_tool_calls(self, tool_calls: list[dict]) -> dict:
        """Process tool calls from AI response."""
        results = {}

        for tool_call in tool_calls:
            function_name = tool_call["function"]["name"]
            arguments = json.loads(tool_call["function"]["arguments"])
            results[function_name] = arguments

        return results

    def _parse_recommendation_from_content(self, content: str, symbol: str) -> TradingRecommendation:
        """Parse trading recommendation from AI response content."""
        # This is a simplified parser - in production, you'd want more robust parsing
        content_lower = content.lower()

        # Determine action
        action = "HOLD"
        if "buy" in content_lower and "sell" not in content_lower:
            action = "BUY"
        elif "sell" in content_lower and "buy" not in content_lower:
            action = "SELL"

        # Extract confidence (simplified)
        confidence = 0.5  # Default
        if "high confidence" in content_lower:
            confidence = 0.8
        elif "low confidence" in content_lower:
            confidence = 0.3

        # Extract risk level
        risk_level = "MEDIUM"
        if "low risk" in content_lower:
            risk_level = "LOW"
        elif "high risk" in content_lower:
            risk_level = "HIGH"

        # Default price range (this would need more sophisticated parsing in practice)
        price_range = (Decimal("0"), Decimal("0"))

        return TradingRecommendation(
            timestamp=datetime.now(),
            symbol=symbol,
            action=action,
            price_range=price_range,
            confidence=confidence,
            reasoning=content[:500],  # Truncate for storage
            risk_level=risk_level
        )

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
