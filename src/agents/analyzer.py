"""Market analyzer agent for data analysis and AI-driven insights."""

import asyncio
import logging
import signal
import sys
from datetime import datetime

from ..core.models import MarketAnalysisResult, TradingRecommendation
from ..core.redis_client import RedisDataStore
from ..utils.ai_client import DeepSeekClient
from ..utils.config import Settings

logger = logging.getLogger(__name__)

# Try to import enhanced analyzer first
try:
    from ..core.analyzers_enhanced import MarketAnalyzer as EnhancedMarketAnalyzer
    logger.info("Using enhanced market analyzer with wave peak detection")
except ImportError:
    from ..core.analyzers import MarketAnalyzer
    logger.warning("Enhanced analyzer not available, falling back to basic analyzer")
    EnhancedMarketAnalyzer = MarketAnalyzer


class AnalyzerAgent:
    """Agent responsible for market analysis and trading recommendations."""

    def __init__(self, settings: Settings):
        """Initialize the analyzer agent."""
        self.settings = settings

        # Initialize components
        self.redis_store = RedisDataStore(
            host=settings.redis.host,
            port=settings.redis.port,
            db=settings.redis.db
        )
        self.market_analyzer = MarketAnalyzer(
            min_volume_threshold=settings.analyzer.analysis.min_order_volume_threshold,
            analysis_window_minutes=180  # 3 hours
        )
        self.ai_client = DeepSeekClient(
            api_key=settings.analyzer.deepseek.api_key,
            base_url=settings.analyzer.deepseek.base_url,
            model=settings.analyzer.deepseek.model,
            max_tokens=settings.analyzer.deepseek.max_tokens,
            temperature=settings.analyzer.deepseek.temperature
        )

        # Control flags
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.is_running = False
        self.shutdown_event.set()

    async def start(self) -> None:
        """Start the analysis process."""
        logger.info("Starting Market Analyzer Agent")

        # Test Redis connection
        if not self.redis_store.test_connection():
            logger.error("Failed to connect to Redis. Exiting...")
            return

        # Main analysis loop
        try:
            self.is_running = True
            await self._analysis_loop()
        except Exception as e:
            logger.error(f"Analyzer agent error: {e}")
        finally:
            await self._shutdown()

    async def _analysis_loop(self) -> None:
        """Main analysis loop."""
        interval = self.settings.analyzer.analysis.interval_seconds

        while self.is_running:
            try:
                logger.debug("Starting market analysis cycle")
                await self._perform_analysis_cycle()

                # Wait for next cycle
                await asyncio.sleep(interval)

            except Exception as e:
                logger.error(f"Analysis cycle error: {e}")
                await asyncio.sleep(10)  # Short delay before retry

    async def _perform_analysis_cycle(self) -> None:
        """Perform a complete analysis cycle."""
        try:
            # Step 1: Get latest depth snapshot
            snapshot = self.redis_store.get_latest_depth_snapshot()
            if not snapshot:
                logger.debug("No depth snapshot available")
                return

            # Step 2: Get recent trade data (last 3 hours for analysis)
            trade_data = self.redis_store.get_recent_trade_data(minutes=180)
            if not trade_data:
                logger.debug("No trade data available")
                return

            logger.info(
                f"Analyzing market data: snapshot from {snapshot.timestamp}, "
                f"{len(trade_data)} minutes of trade data"
            )

            # Step 3: Perform enhanced technical analysis
            analysis_result = self.market_analyzer.analyze_market(
                snapshot=snapshot,
                trade_data_list=trade_data,
                symbol=self.settings.binance.symbol,
                enhanced_mode=True
            )

            # Step 4: Get AI-driven recommendation
            recommendation = await self.ai_client.analyze_market_data(
                analysis_result=analysis_result,
                symbol=self.settings.binance.symbol
            )

            if recommendation:
                # Step 5: Store results
                await self.redis_store.store_analysis_result(analysis_result)

                # Step 6: Log recommendation (for now, until we implement trading)
                await self._log_trading_recommendation(recommendation, analysis_result)

            else:
                logger.warning("AI analysis failed to produce recommendation")

        except Exception as e:
            logger.error(f"Analysis cycle failed: {e}")

    async def _log_trading_recommendation(
        self,
        recommendation: TradingRecommendation,
        analysis_result: MarketAnalysisResult
    ) -> None:
        """Log trading recommendation to file for debugging."""
        try:
            log_entry = {
                'timestamp': recommendation.timestamp.isoformat(),
                'symbol': recommendation.symbol,
                'action': recommendation.action,
                'price_range': [float(recommendation.price_range[0]), float(recommendation.price_range[1])],
                'confidence': recommendation.confidence,
                'reasoning': recommendation.reasoning,
                'risk_level': recommendation.risk_level,
                'market_context': {
                    'support_count': len(analysis_result.support_levels),
                    'resistance_count': len(analysis_result.resistance_levels),
                    'resonance_zones_count': len(analysis_result.resonance_zones),
                    'poc_count': len(analysis_result.poc_levels)
                }
            }

            # Write to log file
            log_file_path = f"logs/trading_recommendations_{recommendation.symbol.lower()}.log"
            import json
            from pathlib import Path

            log_dir = Path(log_file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)

            with open(log_file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            logger.info(
                f"Trading recommendation logged: {recommendation.action} "
                f"with {recommendation.confidence:.2f} confidence"
            )

        except Exception as e:
            logger.error(f"Failed to log trading recommendation: {e}")

    async def _shutdown(self) -> None:
        """Cleanup and shutdown the agent."""
        logger.info("Shutting down Market Analyzer Agent")

        self.is_running = False

        # Close connections
        await self.ai_client.close()
        await self.redis_store.close()

        logger.info("Market Analyzer Agent shutdown complete")

    def get_status(self) -> dict:
        """Get current agent status."""
        return {
            'is_running': self.is_running,
            'redis_connected': self.redis_store.test_connection(),
            'last_analysis': datetime.now().isoformat(),
            'depth_snapshot_available': self.redis_store.depth_snapshot_exists(),
            'trade_window_count': self.redis_store.get_trade_window_count()
        }


async def main() -> None:
    """Main entry point for the analyzer agent."""
    import argparse

    parser = argparse.ArgumentParser(description="Strategy Agent Market Analyzer")
    parser.add_argument(
        "--config",
        default="config/development.yaml",
        help="Configuration file path"
    )
    args = parser.parse_args()

    # Load settings
    settings = Settings.load_from_file(args.config)
    settings.setup_logging()

    # Validate DeepSeek API key
    if not settings.analyzer.deepseek.api_key:
        logger.error("DeepSeek API key is required. Please set DEEPSEEK_API_KEY environment variable.")
        sys.exit(1)

    # Create and start agent
    agent = AnalyzerAgent(settings)

    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
