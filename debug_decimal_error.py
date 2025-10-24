#!/usr/bin/env python3
"""Debug script to isolate the Decimal type error."""

import logging
from decimal import Decimal

from src.core.models import DepthLevel
from src.core.price_aggregator import aggregate_depth_by_one_dollar
from src.core.wave_peak_analyzer import detect_combined_peaks

# Set logging to see errors
logging.basicConfig(level=logging.DEBUG)

def test_aggregation():
    print("Testing 1-dollar aggregation...")

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

    aggregated_bids, aggregated_asks = aggregate_depth_by_one_dollar(bids, asks)
    print(f"Aggregated bids: {aggregated_bids}")
    print(f"Aggregated asks: {aggregated_asks}")

    # Test peak detection on empty data
    print("\nTesting peak detection on empty data...")
    try:
        peaks = detect_combined_peaks({})
        print(f"Empty data peaks: {peaks}")
    except Exception as e:
        print(f"Error with empty data: {e}")
        import traceback
        traceback.print_exc()

    # Test peak detection on aggregated data
    print("\nTesting peak detection on aggregated bids...")
    try:
        peaks = detect_combined_peaks(aggregated_bids)
        print(f"Bids peaks: {peaks}")
    except Exception as e:
        print(f"Error with bids data: {e}")
        import traceback
        traceback.print_exc()

    # Test peak detection on aggregated asks
    print("\nTesting peak detection on aggregated asks...")
    try:
        peaks = detect_combined_peaks(aggregated_asks)
        print(f"Asks peaks: {peaks}")
    except Exception as e:
        print(f"Error with asks data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_aggregation()
