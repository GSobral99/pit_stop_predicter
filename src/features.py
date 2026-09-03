import numpy as np
import pandas as pd

from src.data_loader import load_race

# Typical starting fuel load at the beginning of an F1 race (kg).
# Since there hasn't been refuelling for year I assumed linear
# consumption down to 0 kg on the last lap.
STARTING_FUEL_KG = 110.0


def clean_laps(laps):
    """Applies quality filters to the laps."""
    return laps[
        (laps['IsAccurate'] == True) &
        (laps['PitInTime'].isna()) &
        (laps['PitOutTime'].isna()) &
        (laps['TrackStatus'] == '1')
    ].copy()


def add_laptime_seconds(laps):
    """Converts LapTime (timedelta) to seconds (float)."""
    laps['LapTime_s'] = laps['LapTime'].dt.total_seconds()
    return laps


def add_track_temperature(laps, session):
    """Joins track temperature (TrackTemp) from session.weather_data
    onto each lap, using merge_asof on the nearest timestamp."""
    weather = session.weather_data[['Time', 'TrackTemp', 'AirTemp', 'Humidity']].copy()
    weather = weather.sort_values('Time')

    laps_sorted = laps.sort_values('Time')
    merged = pd.merge_asof(
        laps_sorted,
        weather,
        on='Time',
        direction='nearest',
    )
    return merged.sort_index() if merged.index.is_monotonic_increasing else merged


def add_fuel_load(laps):
    """Estimates fuel load (kg) on each lap, assuming linear consumption
    from STARTING_FUEL_KG down to 0 on the race's last lap."""
    total_laps = laps['LapNumber'].max()
    if pd.isna(total_laps) or total_laps <= 1:
        laps['FuelLoad_kg'] = STARTING_FUEL_KG
        return laps

    consumption_per_lap = STARTING_FUEL_KG / total_laps
    laps['FuelLoad_kg'] = (
        STARTING_FUEL_KG - (laps['LapNumber'] - 1) * consumption_per_lap
    ).clip(lower=0)
    return laps


def build_features(year, gp, session_type='R'):
    session = load_race(year, gp, session_type)
    laps = clean_laps(session.laps)

    laps = add_laptime_seconds(laps)
    laps = add_track_temperature(laps, session)
    laps = add_fuel_load(laps)

    # TyreLife (tyre age in laps) already comes from FastF1.
    feature_cols = [
        'Driver', 'LapNumber', 'Stint', 'Compound', 'TyreLife',
        'LapTime_s', 'TrackTemp', 'AirTemp', 'Humidity', 'FuelLoad_kg',
    ]
    return laps[feature_cols].dropna(subset=['LapTime_s', 'TyreLife', 'Compound'])


def build_dataset(races):
    """races: list of (year, gp, session_type) tuples. Returns a single
    concatenated DataFrame, ready for model training."""
    frames = []
    for year, gp, session_type in races:
        try:
            df = build_features(year, gp, session_type)
            df['Year'] = year
            df['GrandPrix'] = gp
            frames.append(df)
        except Exception as exc:
            print(f'Warning: failed {year} {gp} ({session_type}): {exc}')
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


if __name__ == '__main__':
    df = build_features(2023, 'Monza', 'R')
    print(df.head())
    print(df.describe(include='all'))