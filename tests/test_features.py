import numpy as np
import pandas as pd
import pytest

from src.features import add_fuel_load, add_laptime_seconds, add_track_temperature, clean_laps


def _fake_laps(n=5):
    return pd.DataFrame({
        'IsAccurate': [True] * n,
        'PitInTime': pd.Series([pd.NaT] * n, dtype='timedelta64[ns]'),
        'PitOutTime': pd.Series([pd.NaT] * n, dtype='timedelta64[ns]'),
        'TrackStatus': ['1'] * n,
        'LapNumber': np.arange(1, n + 1),
        'LapTime': pd.to_timedelta([90 + i for i in range(n)], unit='s'),
        'Time': pd.to_timedelta(np.arange(n) * 90, unit='s'),
    })


class TestCleanLaps:
    def test_keeps_valid_laps(self):
        laps = _fake_laps(5)
        result = clean_laps(laps)
        assert len(result) == 5

    def test_drops_inaccurate_laps(self):
        laps = _fake_laps(3)
        laps.loc[1, 'IsAccurate'] = False
        result = clean_laps(laps)
        assert len(result) == 2
        assert False not in result['IsAccurate'].values

    def test_drops_pit_in_and_out_laps(self):
        laps = _fake_laps(4)
        laps.loc[0, 'PitInTime'] = pd.Timedelta(seconds=100)
        laps.loc[1, 'PitOutTime'] = pd.Timedelta(seconds=10)
        result = clean_laps(laps)
        assert len(result) == 2

    def test_drops_non_green_flag_laps(self):
        laps = _fake_laps(3)
        laps.loc[0, 'TrackStatus'] = '4'  # Safety Car, por exemplo
        result = clean_laps(laps)
        assert len(result) == 2

    def test_does_not_mutate_input(self):
        laps = _fake_laps(3)
        original = laps.copy()
        clean_laps(laps)
        pd.testing.assert_frame_equal(laps, original)


class TestLapTimeSeconds:
    def test_converts_timedelta_to_seconds(self):
        laps = _fake_laps(3)
        result = add_laptime_seconds(laps)
        assert result['LapTime_s'].tolist() == [90.0, 91.0, 92.0]

    def test_no_negative_or_nan_for_valid_input(self):
        laps = _fake_laps(5)
        result = add_laptime_seconds(laps)
        assert (result['LapTime_s'] > 0).all()
        assert result['LapTime_s'].notna().all()


class TestFuelLoad:
    def test_starts_near_starting_fuel_on_lap_one(self):
        laps = _fake_laps(10)
        laps['LapNumber'] = np.arange(1, 11)
        result = add_fuel_load(laps)
        assert result['FuelLoad_kg'].iloc[0] == pytest.approx(110.0, abs=1e-6)

    def test_decreases_monotonically(self):
        laps = _fake_laps(20)
        laps['LapNumber'] = np.arange(1, 21)
        result = add_fuel_load(laps)
        diffs = result['FuelLoad_kg'].diff().dropna()
        assert (diffs <= 0).all()

    def test_never_goes_negative(self):
        laps = _fake_laps(5)
        laps['LapNumber'] = np.arange(1, 6)
        result = add_fuel_load(laps)
        assert (result['FuelLoad_kg'] >= 0).all()

    def test_single_lap_race_does_not_crash(self):
        laps = _fake_laps(1)
        laps['LapNumber'] = [1]
        result = add_fuel_load(laps)
        assert result['FuelLoad_kg'].iloc[0] == pytest.approx(110.0)


class _FakeSession:
    def __init__(self, weather_df):
        self.weather_data = weather_df


class TestTrackTemperature:
    def test_merges_nearest_weather_reading(self):
        laps = _fake_laps(3)
        weather = pd.DataFrame({
            'Time': pd.to_timedelta([0, 90, 180], unit='s'),
            'TrackTemp': [30.0, 35.0, 40.0],
            'AirTemp': [20.0, 21.0, 22.0],
            'Humidity': [50.0, 51.0, 52.0],
        })
        session = _FakeSession(weather)
        result = add_track_temperature(laps, session)
        assert result['TrackTemp'].notna().all()
        assert len(result) == len(laps)

    def test_does_not_introduce_extra_rows(self):
        laps = _fake_laps(4)
        weather = pd.DataFrame({
            'Time': pd.to_timedelta([0, 60, 120, 180], unit='s'),
            'TrackTemp': [30.0, 31.0, 32.0, 33.0],
            'AirTemp': [20.0] * 4,
            'Humidity': [50.0] * 4,
        })
        session = _FakeSession(weather)
        result = add_track_temperature(laps, session)
        assert len(result) == len(laps)