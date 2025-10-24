"""Enhanced market analyzers with wave peak detection and 1-dollar precision aggregation.

This module provides sophisticated market analysis using statistical methods
to identify significant price levels and wave patterns.
"""

import logging
from collections import defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Tuple, Optional

from .models import (
    DepthSnapshot,
    EnhancedMarketAnalysisResult,
    MarketAnalysisResult,
    MinuteTradeData,
    SupportResistanceLevel,
)

# Import new modules
try:
    from .price_aggregator import (
        aggregate_depth_by_one_dollar,
        calculate_depth_statistics,
        identify_liquidity_clusters,
        convert_to_depth_levels,
        validate_aggregation_quality,
    )
    from .wave_peak_analyzer import (
        detect_combined_peaks,
        analyze_wave_formation,
        validate_peak_detection_quality,
        WavePeak,
        PriceZone,
    )
except ImportError as e:
    logger.warning(f"Could not import enhanced analyzer modules: {e}")
    # Fallback imports
    from .price_aggregator import (
        aggregate_depth_by_one_dollar,
        calculate_depth_statistics,
        identify_liquidity_clusters,
        convert_to_depth_levels,
        validate_aggregation_quality,
    )
    from .wave_peak_analyzer import (
        detect_combined_peaks,
        analyze_wave_formation,
        validate_peak_detection_quality,
        WavePeak,
        PriceZone,
    )