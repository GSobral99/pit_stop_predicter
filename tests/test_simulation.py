import numpy as np
import pytest

from src.simulation import compare_scenarios, simulate_pit_stop_risk


class TestSimulatePitStopRisk:
    def test_reproducible_with_same_seed(self):
        s1, _ = simulate_pit_stop_risk(n_simulations=500, rng_seed=1)
        s2, _ = simulate_pit_stop_risk(n_simulations=500, rng_seed=1)
        assert s1 == s2

    def test_different_seed_gives_different_result(self):
        s1, _ = simulate_pit_stop_risk(n_simulations=500, rng_seed=1)
        s2, _ = simulate_pit_stop_risk(n_simulations=500, rng_seed=2)
        assert s1 != s2

    def test_probabilities_are_within_0_and_1(self):
        summary, _ = simulate_pit_stop_risk(n_simulations=5000)
        assert 0.0 <= summary['p_gain_position_on_car_ahead'] <= 1.0
        assert 0.0 <= summary['p_lose_position_to_car_behind'] <= 1.0

    def test_dataframe_has_expected_columns_and_length(self):
        n = 1000
        _, df = simulate_pit_stop_risk(n_simulations=n)
        assert len(df) == n
        expected_cols = {
            'safety_car', 'our_pit_loss_s', 'gap_ahead_s', 'gap_behind_s',
            'gained_position_from_ahead', 'lost_position_to_behind',
        }
        assert expected_cols.issubset(df.columns)

    def test_pit_loss_never_negative(self):
        _, df = simulate_pit_stop_risk(n_simulations=5000, pit_loss_mean_s=2.0, pit_loss_std_s=5.0)
        assert (df['our_pit_loss_s'] >= 0).all()

    def test_safety_car_reduces_average_pit_loss(self):
        summary_no_sc, _ = simulate_pit_stop_risk(
            n_simulations=20_000, safety_car_probability=0.0, rng_seed=7
        )
        summary_sc, _ = simulate_pit_stop_risk(
            n_simulations=20_000, safety_car_probability=1.0, rng_seed=7,
            safety_car_pit_loss_s=10.0,
        )
        assert summary_sc['mean_pit_loss_s'] < summary_no_sc['mean_pit_loss_s']

    def test_close_battle_has_more_position_changes_than_clear_air(self):
        # Em ar livre os gaps são maiores -> muito mais difícil o pit
        # stop mudar a ordem, nos dois sentidos.
        close, _ = simulate_pit_stop_risk(n_simulations=20_000, scenario='close_battle', rng_seed=3)
        clear, _ = simulate_pit_stop_risk(n_simulations=20_000, scenario='clear_air', rng_seed=3)
        assert close['p_gain_position_on_car_ahead'] > clear['p_gain_position_on_car_ahead']
        assert close['p_lose_position_to_car_behind'] > clear['p_lose_position_to_car_behind']

    def test_invalid_scenario_raises(self):
        with pytest.raises(ValueError):
            simulate_pit_stop_risk(n_simulations=10, scenario='not_a_real_scenario')

    def test_zero_simulations_returns_empty_dataframe(self):
        summary, df = simulate_pit_stop_risk(n_simulations=0)
        assert len(df) == 0
        # mean() de uma série vazia é NaN — o resumo não deve rebentar,
        # mas o valor não é interpretável; documentamos o caso limite.
        assert summary['n_simulations'] == 0


class TestCompareScenarios:
    def test_returns_four_rows(self):
        result = compare_scenarios(n_simulations=1000)
        assert len(result) == 4
        assert set(result.index) == {
            'close_battle_no_SC', 'close_battle_with_SC',
            'clear_air_no_SC', 'clear_air_with_SC',
        }

    def test_no_sc_rows_have_zero_probability(self):
        result = compare_scenarios(n_simulations=1000)
        assert result.loc['close_battle_no_SC', 'safety_car_probability_used'] == 0.0
        assert result.loc['clear_air_no_SC', 'safety_car_probability_used'] == 0.0

    def test_with_sc_rows_use_requested_probability(self):
        result = compare_scenarios(n_simulations=1000, safety_car_probability_with_sc=0.4)
        assert result.loc['close_battle_with_SC', 'safety_car_probability_used'] == 0.4