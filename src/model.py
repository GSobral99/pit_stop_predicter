import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    _HAS_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    _HAS_XGB = False

from src.features import build_dataset

NUMERIC_FEATURES = ['LapNumber', 'TyreLife', 'TrackTemp', 'AirTemp', 'Humidity', 'FuelLoad_kg']
CATEGORICAL_FEATURES = ['Compound']
TARGET = 'LapTime_s'


def _build_pipeline():
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
    ], remainder='passthrough')

    if _HAS_XGB:
        regressor = XGBRegressor(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
    else:
        regressor = RandomForestRegressor(
            n_estimators=400, max_depth=10, random_state=42, n_jobs=-1
        )

    return Pipeline([
        ('preprocess', preprocessor),
        ('model', regressor),
    ])


import re


def slugify_circuit(gp):
    """'Belgian Grand Prix' -> 'belgian_grand_prix' — used to name the
    per-circuit model file."""
    slug = gp.strip().lower()
    slug = re.sub(r'[^a-z0-9]+', '_', slug)
    return slug.strip('_')


def train_degradation_model(races, test_size=0.2, save_path='models/tyre_degradation.joblib'):
    """races: list of (year, gp, session_type).
    Trains a model that predicts LapTime_s from tyre, fuel, and track
    condition features.

    IMPORTANT: the train/test split is done by RACE (GroupShuffleSplit),
    not by individual lap. A random per-lap split leaves laps from the
    same race in both train and test, causing data leakage: features such
    as FuelLoad_kg (deterministic from LapNumber and total lap count,
    which varies by circuit) and TrackTemp (nearly constant within the
    same race) let the model "identify" the race instead of learning
    tyre degradation in a generic way — inflating R² artificially.
    """
    df = build_dataset(races)
    if df.empty:
        raise ValueError('No data was loaded for training.')

    n_circuits = df['GrandPrix'].nunique()
    if n_circuits > 1:
        print(f'Warning: the training data mixes {n_circuits} different '
              f'circuits ({sorted(df["GrandPrix"].unique().tolist())}). '
              f'A model trained this way tends not to generalize well to '
              f'circuits it has not seen (see earlier negative R2 tests). '
              f'For a per-circuit model, use races from a single GP, '
              f'across several seasons (e.g. 2021, 2022, 2023 of the same '
              f'circuit).')

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    groups = df['Year'].astype(str) + '_' + df['GrandPrix'].astype(str)

    n_groups = groups.nunique()
    if n_groups < 2:
        # Not enough races to split by group; fall back to a random
        # split, but warn that the metrics will be optimistic.
        print('Warning: only 1 race available — falling back to a random '
              'per-lap split (metrics are likely optimistic). Add more '
              'races for a trustworthy evaluation.')
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=groups))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline = _build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    metrics = {
        'MAE_s': mean_absolute_error(y_test, preds),
        'R2': r2_score(y_test, preds),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_groups': n_groups,
        'split_by_race': n_groups >= 2,
        'circuits': sorted(df['GrandPrix'].unique().tolist()),
        'years': sorted(df['Year'].unique().tolist()),
        'engine': 'xgboost' if _HAS_XGB else 'random_forest',
    }

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(pipeline, save_path)

    return pipeline, metrics


def estimate_tyre_advantage_s_per_lap(curve):
    """Estimates the average per-lap pace advantage of a fresh tyre over
    an aging one, from a predicted degradation curve (as returned by
    predict_degradation_curve or found under result['curve']).

    This is exactly the quantity simulate_pit_stop_risk needs for
    `tyre_advantage_s_per_lap`: how many seconds per lap you gain while
    your tyres are fresher than a rival's — but instead of a fixed guess,
    it comes straight from the trained model's own prediction for this
    circuit and compound.
    """
    if len(curve) < 2:
        return 0.0
    total_laps = len(curve)
    total_degradation = curve['PredictedLapTime_s'].iloc[-1] - curve['PredictedLapTime_s'].iloc[0]
    return max(0.0, total_degradation / (total_laps - 1))


def predict_degradation_curve(pipeline, compound, track_temp, fuel_start_kg,
                               total_laps, air_temp=25.0, humidity=50.0):
    """Predicts LapTime_s over the tyre's life (0..total_laps-1),
    assuming linear fuel consumption."""
    tyre_life = np.arange(0, total_laps)
    fuel = np.clip(fuel_start_kg - tyre_life * (fuel_start_kg / total_laps), 0, None)

    curve = pd.DataFrame({
        'LapNumber': tyre_life + 1,
        'TyreLife': tyre_life,
        'TrackTemp': track_temp,
        'AirTemp': air_temp,
        'Humidity': humidity,
        'FuelLoad_kg': fuel,
        'Compound': compound,
    })
    curve['PredictedLapTime_s'] = pipeline.predict(curve[NUMERIC_FEATURES + CATEGORICAL_FEATURES])
    return curve


def _stint_predictions(pipeline, compound, track_temp, fuel_start_kg, total_laps,
                        start_lap, n_laps, air_temp, humidity):
    """Predicts LapTime_s for a stint of `n_laps` laps starting at the
    absolute lap `start_lap` (1-indexed) on a fresh tyre (TyreLife=0).
    FuelLoad_kg always uses the race's ABSOLUTE lap number, not the
    stint's."""
    tyre_life = np.arange(0, n_laps)
    lap_numbers = start_lap + tyre_life
    fuel = np.clip(
        fuel_start_kg - (lap_numbers - 1) * (fuel_start_kg / total_laps), 0, None
    )
    stint_df = pd.DataFrame({
        'LapNumber': lap_numbers,
        'TyreLife': tyre_life,
        'TrackTemp': track_temp,
        'AirTemp': air_temp,
        'Humidity': humidity,
        'FuelLoad_kg': fuel,
        'Compound': compound,
    })
    stint_df['PredictedLapTime_s'] = pipeline.predict(
        stint_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    )
    return stint_df


def suggest_pit_window(pipeline, compound, track_temp, fuel_start_kg, total_laps,
                        pit_loss_s=22.0, second_compound=None, min_stint_laps=5,
                        air_temp=25.0, humidity=50.0):
    """Suggests the ideal pit stop lap (one-stop strategy): tests each
    candidate lap as a stop point, adds up the predicted 1st-stint time +
    pit_loss_s + predicted 2nd-stint time (fresh tyre), and picks the lap
    that minimizes total race time.

    This replaces an arbitrary "accumulated loss" threshold with a direct
    strategy comparison — more realistic, and it always returns an answer
    (it can never come back empty)."""
    second_compound = second_compound or compound
    min_stint_laps = max(1, min_stint_laps)

    if total_laps < 2 * min_stint_laps:
        raise ValueError(
            f'total_laps ({total_laps}) is too short for two stints of at '
            f'least {min_stint_laps} laps each.'
        )

    candidate_laps = range(min_stint_laps, total_laps - min_stint_laps + 1)
    totals = {}
    for pit_lap in candidate_laps:
        stint1 = _stint_predictions(
            pipeline, compound, track_temp, fuel_start_kg, total_laps,
            start_lap=1, n_laps=pit_lap, air_temp=air_temp, humidity=humidity,
        )
        stint2 = _stint_predictions(
            pipeline, second_compound, track_temp, fuel_start_kg, total_laps,
            start_lap=pit_lap + 1, n_laps=total_laps - pit_lap,
            air_temp=air_temp, humidity=humidity,
        )
        total_time = (
            stint1['PredictedLapTime_s'].sum()
            + stint2['PredictedLapTime_s'].sum()
            + pit_loss_s
        )
        totals[pit_lap] = total_time

    optimal_lap = min(totals, key=totals.get)

    # Reference curve (single stint, for inspection/plots).
    curve = predict_degradation_curve(
        pipeline, compound, track_temp, fuel_start_kg, total_laps,
        air_temp=air_temp, humidity=humidity,
    )

    return {
        'optimal_pit_lap': optimal_lap,
        'total_time_by_pit_lap': totals,
        'curve': curve,
    }


if __name__ == '__main__':
    races = [
        (2023, 'Monza', 'R'),
        (2023, 'Spa', 'R'),
        (2023, 'Silverstone', 'R'),
    ]
    pipeline, metrics = train_degradation_model(races)
    print('Model metrics:', metrics)

    result = suggest_pit_window(
        pipeline, compound='MEDIUM', track_temp=35.0,
        fuel_start_kg=110.0, total_laps=53,
    )
    print('Optimal pit stop lap:', result['optimal_pit_lap'])