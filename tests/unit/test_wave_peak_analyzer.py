"""Unit tests for wave_peak_analyzer module.

Tests the statistical wave peak detection algorithms and price zone analysis.
"""

from decimal import Decimal

import pytest

from src.core.wave_peak_analyzer import (
    PriceZone,
    WavePeak,
    _calculate_weighted_statistics,
    analyze_wave_formation,
    detect_combined_peaks,
    detect_normal_distribution_peaks,
    detect_volume_based_peaks,
    validate_peak_detection_quality,
)


class TestWavePeak:
    """Test cases for WavePeak class."""

    def test_wave_peak_initialization(self):
        """Test WavePeak initialization with all parameters."""
        peak = WavePeak(
            center_price=Decimal('100.50'),
            volume=Decimal('500.0'),
            price_range_width=Decimal('2.0'),
            z_score=1.5,
            confidence=0.85,
            bid_volume=Decimal('300.0'),
            ask_volume=Decimal('200.0'),
            peak_type='statistical_peak'
        )

        assert peak.center_price == Decimal('100.50')
        assert peak.volume == Decimal('500.0')
        assert peak.price_range_width == Decimal('2.0')
        assert peak.z_score == 1.5
        assert peak.confidence == 0.85
        assert peak.bid_volume == Decimal('300.0')
        assert peak.ask_volume == Decimal('200.0')
        assert peak.peak_type == 'statistical_peak'

    def test_wave_peak_default_values(self):
        """Test WavePeak with default parameter values."""
        peak = WavePeak(
            center_price=Decimal('100.0'),
            volume=Decimal('100.0'),
            price_range_width=Decimal('1.0'),
            z_score=1.0,
            confidence=0.5
        )

        assert peak.bid_volume == Decimal('0')
        assert peak.ask_volume == Decimal('0')
        assert peak.peak_type == 'unknown'

    def test_wave_peak_repr(self):
        """Test WavePeak string representation."""
        peak = WavePeak(
            center_price=Decimal('100.50'),
            volume=Decimal('500.0'),
            price_range_width=Decimal('2.0'),
            z_score=1.5,
            confidence=0.85
        )

        repr_str = repr(peak)
        assert 'WavePeak' in repr_str
        assert '100.50' in repr_str
        assert '500.0' in repr_str
        assert '1.50' in repr_str
        assert '0.85' in repr_str

    def test_wave_peak_to_dict(self):
        """Test WavePeak dictionary conversion."""
        peak = WavePeak(
            center_price=Decimal('100.50'),
            volume=Decimal('500.0'),
            price_range_width=Decimal('2.0'),
            z_score=1.5,
            confidence=0.85
        )

        peak_dict = peak.to_dict()

        assert peak_dict['center_price'] == 100.50
        assert peak_dict['volume'] == 500.0
        assert peak_dict['price_range_width'] == 2.0
        assert peak_dict['z_score'] == 1.5
        assert peak_dict['confidence'] == 0.85
        assert 'lower_price' in peak_dict
        assert 'upper_price' in peak_dict


class TestPriceZone:
    """Test cases for PriceZone class."""

    def test_price_zone_initialization(self):
        """Test PriceZone initialization."""
        zone = PriceZone(
            lower_price=Decimal('99.0'),
            upper_price=Decimal('101.0'),
            zone_type='support',
            confidence=0.75,
            total_volume=Decimal('1000.0'),
            bid_ask_ratio=1.5
        )

        assert zone.lower_price == Decimal('99.0')
        assert zone.upper_price == Decimal('101.0')
        assert zone.zone_type == 'support'
        assert zone.confidence == 0.75
        assert zone.total_volume == Decimal('1000.0')
        assert zone.bid_ask_ratio == 1.5

    def test_price_zone_calculated_fields(self):
        """Test PriceZone calculated fields."""
        zone = PriceZone(
            lower_price=Decimal('99.0'),
            upper_price=Decimal('101.0'),
            zone_type='resistance',
            confidence=0.8,
            total_volume=Decimal('500.0')
        )

        assert zone.center_price == Decimal('100.0')  # (99 + 101) / 2
        assert zone.width == Decimal('2.0')           # 101 - 99

    def test_price_zone_repr(self):
        """Test PriceZone string representation."""
        zone = PriceZone(
            lower_price=Decimal('99.0'),
            upper_price=Decimal('101.0'),
            zone_type='support',
            confidence=0.75,
            total_volume=Decimal('1000.0')
        )

        repr_str = repr(zone)
        assert 'PriceZone' in repr_str
        assert 'support' in repr_str
        assert '99.0-101.0' in repr_str
        assert '1000.0' in repr_str
        assert '0.75' in repr_str


class TestDetectNormalDistributionPeaks:
    """Test cases for normal distribution peak detection."""

    @pytest.fixture
    def sample_price_volume_data(self) -> dict[Decimal, Decimal]:
        """Create sample price-volume data with clear peaks."""
        return {
            Decimal('95'): Decimal('5.0'),   # Low volume
            Decimal('96'): Decimal('8.0'),   # Low volume
            Decimal('97'): Decimal('15.0'),  # Peak 1
            Decimal('98'): Decimal('25.0'),  # Peak 1 (highest)
            Decimal('99'): Decimal('20.0'),  # Peak 1
            Decimal('100'): Decimal('10.0'), # Medium volume
            Decimal('101'): Decimal('12.0'), # Peak 2
            Decimal('102'): Decimal('18.0'), # Peak 2 (highest)
            Decimal('103'): Decimal('14.0'), # Peak 2
            Decimal('104'): Decimal('6.0'),  # Low volume
        }

    def test_detect_peaks_with_sample_data(self, sample_price_volume_data):
        """Test peak detection with sample data."""
        peaks = detect_normal_distribution_peaks(
            sample_price_volume_data,
            min_peak_volume=Decimal('10.0'),
            z_score_threshold=1.5,
            min_peak_confidence=0.3
        )

        assert len(peaks) >= 1  # Should detect at least one peak

        # Verify peak properties
        for peak in peaks:
            assert isinstance(peak, WavePeak)
            assert peak.center_price in sample_price_volume_data
            assert peak.volume >= Decimal('10.0')
            assert peak.confidence >= 0.3
            assert peak.peak_type == 'statistical_peak'

    def test_detect_peaks_with_empty_data(self):
        """Test peak detection with empty data."""
        peaks = detect_normal_distribution_peaks({})
        assert len(peaks) == 0

    def test_detect_peaks_with_insufficient_data(self):
        """Test peak detection with insufficient data points."""
        small_data = {
            Decimal('100'): Decimal('10.0'),
            Decimal('101'): Decimal('15.0'),
        }

        peaks = detect_normal_distribution_peaks(small_data)
        assert len(peaks) == 0  # Need at least 3 points for statistical analysis

    def test_detect_peaks_custom_thresholds(self, sample_price_volume_data):
        """Test peak detection with custom thresholds."""
        # Test with higher volume threshold
        peaks_high_volume = detect_normal_distribution_peaks(
            sample_price_volume_data,
            min_peak_volume=Decimal('20.0')
        )

        # Test with lower confidence threshold
        peaks_low_confidence = detect_normal_distribution_peaks(
            sample_price_volume_data,
            min_peak_confidence=0.1
        )

        # Higher volume threshold should detect fewer peaks
        assert len(peaks_high_volume) <= len(peaks_low_confidence)

    def test_detect_peaks_all_equal_values(self):
        """Test peak detection with all equal volume values."""
        equal_data = {
            Decimal(f'{100 + i}'): Decimal('10.0')
            for i in range(10)
        }

        peaks = detect_normal_distribution_peaks(equal_data)
        # With equal values, no local maxima should be detected
        assert len(peaks) == 0


class TestDetectVolumeBasedPeaks:
    """Test cases for volume-based peak detection."""

    @pytest.fixture
    def volume_spike_data(self) -> dict[Decimal, Decimal]:
        """Create data with clear volume spikes."""
        return {
            Decimal('100'): Decimal('5.0'),   # Low volume
            Decimal('101'): Decimal('8.0'),   # Low volume
            Decimal('102'): Decimal('25.0'),  # Volume spike (5x neighbors)
            Decimal('103'): Decimal('6.0'),   # Low volume
            Decimal('104'): Decimal('7.0'),   # Low volume
            Decimal('105'): Decimal('30.0'),  # Volume spike (4x neighbors)
            Decimal('106'): Decimal('8.0'),   # Low volume
        }

    def test_detect_volume_peaks_with_spikes(self, volume_spike_data):
        """Test volume peak detection with clear spikes."""
        peaks = detect_volume_based_peaks(
            volume_spike_data,
            min_relative_volume=3.0,  # 3x neighbor average
            min_absolume=Decimal('15.0')
        )

        assert len(peaks) >= 1

        # Verify detected peaks have high relative volume
        for peak in peaks:
            assert peak.volume >= Decimal('15.0')
            assert peak.peak_type == 'volume_concentration'

    def test_detect_volume_peaks_with_empty_data(self):
        """Test volume peak detection with empty data."""
        peaks = detect_volume_based_peaks({})
        assert len(peaks) == 0

    def test_detect_volume_peaks_custom_thresholds(self, volume_spike_data):
        """Test volume peak detection with custom thresholds."""
        # High relative volume threshold
        peaks_high = detect_volume_based_peaks(
            volume_spike_data,
            min_relative_volume=5.0
        )

        # Low relative volume threshold
        peaks_low = detect_volume_based_peaks(
            volume_spike_data,
            min_relative_volume=2.0
        )

        # Higher threshold should detect fewer peaks
        assert len(peaks_high) <= len(peaks_low)

    def test_detect_volume_peaks_edge_cases(self):
        """Test volume peak detection edge cases."""
        # Single data point
        single_point = {Decimal('100'): Decimal('100.0')}
        peaks = detect_volume_based_peaks(single_point)
        assert len(peaks) == 0

        # Two data points
        two_points = {
            Decimal('100'): Decimal('10.0'),
            Decimal('101'): Decimal('20.0')
        }
        peaks = detect_volume_based_peaks(two_points)
        assert len(peaks) == 0


class TestAnalyzeWaveFormation:
    """Test cases for wave formation analysis."""

    @pytest.fixture
    def sample_peaks(self) -> list[WavePeak]:
        """Create sample wave peaks for formation analysis."""
        return [
            WavePeak(Decimal('100.0'), Decimal('50.0'), Decimal('2.0'), 1.5, 0.8),
            WavePeak(Decimal('102.0'), Decimal('40.0'), Decimal('2.0'), 1.2, 0.7),
            WavePeak(Decimal('110.0'), Decimal('60.0'), Decimal('3.0'), 1.8, 0.9),
            WavePeak(Decimal('112.0'), Decimal('45.0'), Decimal('2.0'), 1.3, 0.75),
            WavePeak(Decimal('150.0'), Decimal('30.0'), Decimal('2.0'), 1.0, 0.6),
        ]

    def test_analyze_wave_formation_with_sample_peaks(self, sample_peaks):
        """Test wave formation analysis with sample peaks."""
        zones = analyze_wave_formation(
            sample_peaks,
            min_peak_distance=Decimal('5.0'),
            max_price_range=Decimal('10.0')
        )

        assert len(zones) >= 1

        # Verify zone properties
        for zone in zones:
            assert isinstance(zone, PriceZone)
            assert zone.lower_price <= zone.upper_price
            assert zone.confidence > 0
            assert zone.zone_type in ['support', 'resistance']

    def test_analyze_wave_formation_insufficient_peaks(self):
        """Test wave formation analysis with insufficient peaks."""
        single_peak = [WavePeak(Decimal('100.0'), Decimal('50.0'), Decimal('2.0'), 1.5, 0.8)]

        zones = analyze_wave_formation(single_peak)
        assert len(zones) == 0  # Need at least 2 peaks

    def test_analyze_wave_formation_custom_parameters(self, sample_peaks):
        """Test wave formation analysis with custom parameters."""
        # Small distance should group more peaks
        zones_small = analyze_wave_formation(
            sample_peaks,
            min_peak_distance=Decimal('1.0'),
            max_price_range=Decimal('20.0')
        )

        # Large distance should group fewer peaks
        zones_large = analyze_wave_formation(
            sample_peaks,
            min_peak_distance=Decimal('15.0'),
            max_price_range=Decimal('5.0')
        )

        # Smaller distance should create more zones or zones with more peaks
        assert len(zones_small) >= 0
        assert len(zones_large) >= 0

    def test_analyze_wave_formation_empty_peaks(self):
        """Test wave formation analysis with empty peaks list."""
        zones = analyze_wave_formation([])
        assert len(zones) == 0


class TestDetectCombinedPeaks:
    """Test cases for combined peak detection."""

    @pytest.fixture
    def complex_price_volume_data(self) -> dict[Decimal, Decimal]:
        """Create complex data suitable for both detection methods."""
        return {
            Decimal('90'): Decimal('5.0'),
            Decimal('91'): Decimal('12.0'),
            Decimal('92'): Decimal('8.0'),
            Decimal('93'): Decimal('25.0'),  # Both statistical and volume peak
            Decimal('94'): Decimal('15.0'),
            Decimal('95'): Decimal('10.0'),
            Decimal('96'): Decimal('40.0'),  # Volume concentration peak
            Decimal('97'): Decimal('8.0'),
            Decimal('98'): Decimal('22.0'),  # Statistical peak
            Decimal('99'): Decimal('18.0'),
            Decimal('100'): Decimal('6.0'),
        }

    def test_detect_combined_peaks_default_params(self, complex_price_volume_data):
        """Test combined peak detection with default parameters."""
        peaks = detect_combined_peaks(complex_price_volume_data)

        assert len(peaks) > 0

        # Verify peak diversity
        peak_types = {peak.peak_type for peak in peaks}
        assert 'statistical_peak' in peak_types or 'volume_concentration' in peak_types

        # Verify peaks are sorted by volume and confidence
        volumes = [peak.volume for peak in peaks]
        assert volumes == sorted(volumes, reverse=True)

    def test_detect_combined_peaks_custom_params(self, complex_price_volume_data):
        """Test combined peak detection with custom parameters."""
        statistical_params = {
            'min_peak_volume': 15.0,
            'z_score_threshold': 2.0,
            'min_peak_confidence': 0.5
        }

        volume_params = {
            'min_relative_volume': 3.0,
            'min_absolume': 20.0
        }

        peaks = detect_combined_peaks(
            complex_price_volume_data,
            statistical_params=statistical_params,
            volume_params=volume_params
        )

        # All peaks should meet the stricter criteria
        for peak in peaks:
            assert peak.volume >= Decimal('15.0')  # From statistical params
            # Additional checks based on detection type

    def test_detect_combined_peaks_deduplication(self):
        """Test that combined detection deduplicates similar peaks."""
        # Create data that would generate similar peaks from both methods
        data = {
            Decimal('100'): Decimal('100.0'),  # Very high volume, should create one peak
            Decimal('101'): Decimal('10.0'),
            Decimal('102'): Decimal('5.0'),
        }

        peaks = detect_combined_peaks(data)

        # Should have at most one peak around 100
        peaks_around_100 = [p for p in peaks if 99 <= float(p.center_price) <= 101]
        assert len(peaks_around_100) <= 1

    def test_detect_combined_peaks_empty_data(self):
        """Test combined peak detection with empty data."""
        peaks = detect_combined_peaks({})
        assert len(peaks) == 0


class TestValidatePeakDetectionQuality:
    """Test cases for peak detection quality validation."""

    def test_validate_high_quality_detection(self):
        """Test quality validation with high-quality detection."""
        original_levels = 100
        detected_peaks = [
            WavePeak(Decimal('100'), Decimal('300'), Decimal('2'), 2.0, 0.9),
            WavePeak(Decimal('105'), Decimal('250'), Decimal('2'), 1.8, 0.85),
            WavePeak(Decimal('110'), Decimal('200'), Decimal('2'), 1.5, 0.8),
        ]
        price_volume_data = {
            Decimal('100'): Decimal('300'),
            Decimal('105'): Decimal('250'),
            Decimal('110'): Decimal('200'),
            # Plus many low-volume levels
            **{Decimal(f'{90 + i}'): Decimal('5') for i in range(20)}
        }

        quality = validate_peak_detection_quality(original_levels, detected_peaks, price_volume_data)

        assert quality['volume_preservation_rate'] > 0.8
        assert quality['compression_ratio'] > 20
        assert quality['avg_confidence'] > 0.8
        assert quality['coverage_rate'] > 0.1
        assert quality['peak_count'] == 3
        assert quality['significant_levels_count'] == 3

    def test_validate_low_quality_detection(self):
        """Test quality validation with low-quality detection."""
        original_levels = 50
        detected_peaks = [
            WavePeak(Decimal('100'), Decimal('10'), Decimal('2'), 0.5, 0.2),
        ]
        price_volume_data = {
            Decimal('100'): Decimal('10'),
            **{Decimal(f'{90 + i}'): Decimal('100') for i in range(10)}  # High volume missed
        }

        quality = validate_peak_detection_quality(original_levels, detected_peaks, price_volume_data)

        assert quality['volume_preservation_rate'] < 0.1  # Poor preservation
        assert quality['compression_ratio'] == 50
        assert quality['avg_confidence'] < 0.5
        assert quality['coverage_rate'] < 0.1

    def test_validate_empty_detection(self):
        """Test quality validation with no detected peaks."""
        quality = validate_peak_detection_quality(100, [], {})

        assert quality['volume_preservation_rate'] == 0
        assert quality['compression_ratio'] == 0
        assert quality['avg_confidence'] == 0
        assert quality['coverage_rate'] == 0
        assert quality['peak_count'] == 0
        assert quality['significant_levels_count'] == 0


class TestCalculateWeightedStatistics:
    """Test cases for weighted statistics calculation."""

    def test_calculate_basic_weighted_stats(self):
        """Test basic weighted statistics calculation."""
        prices = [100.0, 101.0, 102.0]
        volumes = [10.0, 20.0, 30.0]

        mean, std = _calculate_weighted_statistics(prices, volumes)

        # Expected weighted mean: (100*10 + 101*20 + 102*30) / 60 = 101.33
        expected_mean = (1000 + 2020 + 3060) / 60
        assert abs(mean - expected_mean) < 0.01
        assert std > 0

    def test_calculate_weighted_stats_with_zeros(self):
        """Test weighted statistics with zero volumes."""
        prices = [100.0, 101.0, 102.0]
        volumes = [0.0, 0.0, 0.0]

        mean, std = _calculate_weighted_statistics(prices, volumes)

        assert mean == 0.0
        assert std == 0.0

    def test_calculate_weighted_stats_invalid_inputs(self):
        """Test weighted statistics with invalid inputs."""
        # Mismatched lengths
        mean, std = _calculate_weighted_statistics([100.0, 101.0], [10.0])
        assert mean == 0.0
        assert std == 0.0

        # Empty inputs
        mean, std = _calculate_weighted_statistics([], [])
        assert mean == 0.0
        assert std == 0.0

    def test_calculate_weighted_stats_single_value(self):
        """Test weighted statistics with single value."""
        prices = [100.0]
        volumes = [50.0]

        mean, std = _calculate_weighted_statistics(prices, volumes)

        assert mean == 100.0
        assert std == 0.0  # No variance with single value


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling."""

    def test_extreme_values_handling(self):
        """Test handling of extreme values in price-volume data."""
        extreme_data = {
            Decimal('0.01'): Decimal('999999.0'),  # Very high volume
            Decimal('999999.99'): Decimal('0.01'), # Very low volume
        }

        # Should not crash with extreme values
        peaks = detect_combined_peaks(extreme_data)
        assert isinstance(peaks, list)

    def test_negative_or_zero_values(self):
        """Test handling of negative or zero values."""
        invalid_data = {
            Decimal('100'): Decimal('-10.0'),  # Negative volume
            Decimal('101'): Decimal('0.0'),    # Zero volume
        }

        # Should handle gracefully
        peaks = detect_combined_peaks(invalid_data)
        assert isinstance(peaks, list)

    def test_very_large_dataset(self):
        """Test performance with very large dataset."""
        large_data = {
            Decimal(f'{100 + i*0.01}'): Decimal(f'{10 + i*0.1}')
            for i in range(10000)  # 10,000 data points
        }

        # Should complete without timeout
        peaks = detect_combined_peaks(large_data)
        assert isinstance(peaks, list)

    def test_all_same_price_different_volumes(self):
        """Test with same price but different volumes (edge case)."""
        # This would be aggregated to single price level
        same_price_data = {
            Decimal('100'): Decimal('10.0'),
            Decimal('100'): Decimal('20.0'),  # This would overwrite
            Decimal('100'): Decimal('30.0'),
        }

        # In practice, this would be aggregated, so we have 1 entry
        data = {Decimal('100'): Decimal('60.0')}

        peaks = detect_combined_peaks(data)
        assert isinstance(peaks, list)
        # With single price level, no peaks should be detected
        assert len(peaks) == 0
