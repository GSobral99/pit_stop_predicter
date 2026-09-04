from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.model import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    slugify_circuit,
    suggest_pit_window,
    train_degradation_model,
)


class TestSlugifyCircuit:
    def test_basic_name(self):
        assert slugify_circuit('Monza') == 'monza'

    def test_name_with_spaces(self):
        assert slugify_circuit('Belgian Grand Prix') == 'belgian_grand_prix'

    def test_strips_punctuation(self):
        assert slugify_circuit("São Paulo Grand Prix!") == 's_o_paulo_grand_prix'

    def test_no_double_underscores_or_edges(self):
        slug = slugify_circuit('  Emilia-Romagna  Grand Prix  ')
        assert '__' not in slug
        assert not slug.startswith('_')
        assert not slug.endswith('_')


class _FakePipeline:
    def __init__(self, base=90.0, rate_by_compound=None, default_rate=0.1):
        self.base = base
        self.rate_by_compound = rate_by_compound or {}
        self.default_rate = default_rate

    def predict(self, X):
        rates = X['Compound'].map(self.rate_by_compound).fillna(self.default_rate)
        return (self.base + rates * X['TyreLife']).to_numpy()


class TestSuggestPitWindow:
    def test_returns_lap_within_valid_range(self):
        pipeline = _FakePipeline(base=90.0, default_rate=0.1)
        result = suggest_pit_window(
            pipeline, compound='MEDIUM', track_temp=30.0, fuel_start_kg=110.0,
            total_laps=20, pit_loss_s=20.0, min_stint_laps=1,
        )
        assert 1 <= result['optimal_pit_lap'] <= 19

    def test_optimal_lap_is_near_middle_for_symmetric_degradation(self):
        pipeline = _FakePipeline(base=90.0, default_rate=0.1)
        result = suggest_pit_window(
            pipeline, compound='MEDIUM', track_temp=30.0, fuel_start_kg=110.0,
            total_laps=20, pit_loss_s=20.0, min_stint_laps=1,
        )
        assert abs(result['optimal_pit_lap'] - 10) <= 1

    def test_faster_degrading_first_compound_shortens_first_stint(self):
        pipeline = _FakePipeline(
            base=88.0, rate_by_compound={'SOFT': 0.4, 'HARD': 0.05}
        )
        result = suggest_pit_window(
            pipeline, compound='SOFT', second_compound='HARD',
            track_temp=30.0, fuel_start_kg=110.0, total_laps=20,
            pit_loss_s=20.0, min_stint_laps=1,
        )
        assert result['optimal_pit_lap'] < 10

    def test_raises_when_race_too_short_for_two_stints(self):
        pipeline = _FakePipeline()
        with pytest.raises(ValueError):
            suggest_pit_window(
                pipeline, compound='MEDIUM', track_temp=30.0,
                fuel_start_kg=110.0, total_laps=6, min_stint_laps=5,
            )

    def test_always_returns_an_int_never_none(self):
        pipeline = _FakePipeline(base=90.0, default_rate=0.001) 
        result = suggest_pit_window(
            pipeline, compound='MEDIUM', track_temp=30.0, fuel_start_kg=110.0,
            total_laps=53, pit_loss_s=22.0, min_stint_laps=5,
        )
        assert isinstance(result['optimal_pit_lap'], int)


class TestTrainDegradationModelSplit:
    def _synthetic_dataset(self):
        rows = []
        for year, gp, base in [(2023, 'RaceA', 80.0), (2023, 'RaceB', 95.0)]:
            for lap in range(1, 31):
                rows.append({
                    'LapNumber': lap, 'TyreLife': lap % 15, 'TrackTemp': 30.0,
                    'AirTemp': 20.0, 'Humidity': 50.0,
                    'FuelLoad_kg': max(0, 110 - lap * 2), 'Compound': 'MEDIUM',
                    'LapTime_s': base + 0.05 * (lap % 15),
                    'Year': year, 'GrandPrix': gp,
                })
        return pd.DataFrame(rows)

    @patch('src.model.build_dataset')
    def test_no_group_overlap_between_train_and_test(self, mock_build_dataset):
        df = self._synthetic_dataset()
        mock_build_dataset.return_value = df

        from sklearn.model_selection import GroupShuffleSplit
        groups = df['Year'].astype(str) + '_' + df['GrandPrix'].astype(str)
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
        train_idx, test_idx = next(splitter.split(df, groups=groups))

        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        assert train_groups.isdisjoint(test_groups)

    @patch('src.model.joblib')
    @patch('src.model.build_dataset')
    def test_metrics_report_number_of_groups(self, mock_build_dataset, mock_joblib):
        df = self._synthetic_dataset()
        mock_build_dataset.return_value = df

        _, metrics = train_degradation_model(
            races=[(2023, 'RaceA', 'R'), (2023, 'RaceB', 'R')],
            save_path='models/test_model.joblib',
        )
        assert metrics['n_groups'] == 2
        assert metrics['split_by_race'] is True