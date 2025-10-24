#!/usr/bin/env python3
"""Debug script to trace the enhanced analyzer error."""

import logging
from datetime import datetime
from decimal import Decimal

from src.core.analyzers_enhanced import EnhancedMarketAnalyzer
from src.core.models import DepthLevel, DepthSnapshot

# Set logging to see all errors
logging.basicConfig(level=logging.DEBUG)

def test_enhanced_analyzer():
    print("Testing Enhanced Market Analyzer...")

    # Test data
    bids = [
        DepthLevel(Decimal('99.50'), Decimal('10.0')),
        DepthLevel(Decimal('99.20'), Decimal('5.0')),
        DepthLevel(Decimal('98.90'), Decimal('8.0')),
        DepthLevel(Decimal('99.10'), Decimal('12.0')),
    ]

    asks = [
        DepthLevel(Decimal('100.10'), Decimal('8.0')),
        DepthLevel(Decimal('100.50'), Decimal('6.0')),
        DepthLevel(Decimal('101.30'), Decimal('4.0')),
    ]

    snapshot = DepthSnapshot('BTCFDUSD', datetime.now(), bids, asks)

    # Create analyzer
    analyzer = EnhancedMarketAnalyzer(
        min_volume_threshold=Decimal('1.0'),
        analysis_window_minutes=180
    )

    print("Analyzing market...")
    try:
        result = analyzer.analyze_market(
            snapshot=snapshot,
            trade_data_list=[],  # Empty trade data
            symbol='BTCFDUSD',
            enhanced_mode=True
        )
        print(f"Analysis successful: {result}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_enhanced_analyzer()
