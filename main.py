"""
Full pipeline for the Predictive Pit Stop Strategy & Tyre Degradation Simulator.

Connects the four modules:
    data_loader.py -> features.py -> model.py -> simulation.py

Per-circuit model (recommended): train with several seasons of the SAME
GP, so the model learns that circuit's degradation well instead of trying
(and failing) to generalize across different circuits — see the README /
the discussion about negative R2 when mixing circuits.

Usage:
    python main.py
    python main.py --gp Monza --years 2021 2022 2023
    python main.py --gp "Belgian Grand Prix" --years 2022 2023 \
                    --compound MEDIUM --track-temp 35 --total-laps 44 \
                    --pit-loss 22 --sc-probability 0.3

If no races are given, a sample set is used (Monza, 2021-2023). Training
requires access to the FastF1 API (not available in every environment) —
if it fails, the script explains the error and exits cleanly.
"""
import argparse
import sys

from src.model import (
    estimate_tyre_advantage_s_per_lap,
    slugify_circuit,
    suggest_pit_window,
    train_degradation_model,
)
from src.simulation import compare_scenarios


DEFAULT_RACES_YEARS = [2021, 2022, 2023]
DEFAULT_RACES_GP = ['Monza']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pipeline: FastF1 data -> features -> degradation model -> Monte Carlo simulation.'
    )
    parser.add_argument('--years', nargs='+', type=int, default=DEFAULT_RACES_YEARS,
                         help='Season(s) used to train the model.')
    parser.add_argument('--gp', nargs='+', default=DEFAULT_RACES_GP,
                         help="Grand(s) Prix used to train the model. For a "
                              "per-circuit model (recommended), use a single "
                              "GP with several --years.")
    parser.add_argument('--session', default='R',
                         help="FastF1 session type (default: 'R' = race).")

    parser.add_argument('--compound', default='MEDIUM',
                         choices=['SOFT', 'MEDIUM', 'HARD'],
                         help='Tyre compound for the degradation prediction.')
    parser.add_argument('--track-temp', type=float, default=35.0,
                         help='Track temperature (°C) used in the prediction.')
    parser.add_argument('--total-laps', type=int, default=53,
                         help='Total number of laps of the race being simulated.')
    parser.add_argument('--fuel-start', type=float, default=110.0,
                         help='Starting fuel load (kg) used in the prediction.')

    parser.add_argument('--pit-loss', type=float, default=22.0,
                         help='Average time lost in a normal pit stop (s).')
    parser.add_argument('--sc-probability', type=float, default=0.3,
                         help='Safety Car probability used in the "with SC" scenario.')
    parser.add_argument('--n-simulations', type=int, default=20_000,
                         help='Number of Monte Carlo simulations per scenario.')

    parser.add_argument('--model-path', default=None,
                         help='Path to save the trained model. By default, it '
                              "is generated automatically from the circuit(s) "
                              "(e.g. models/tyre_degradation_monza.joblib).")

    return parser.parse_args()


def _default_model_path(gps):
    if len(gps) == 1:
        return f'models/tyre_degradation_{slugify_circuit(gps[0])}.joblib'
    return f'models/tyre_degradation_multi_{"_".join(slugify_circuit(g) for g in gps)}.joblib'


def run_pipeline(args):
    races = [(year, gp, args.session) for year in args.years for gp in args.gp]
    model_path = args.model_path or _default_model_path(args.gp)

    if len(args.gp) == 1:
        print(f"\nMode: per-circuit model ('{args.gp[0]}'), seasons: {args.years}")
    else:
        print(f"\nWarning: training with {len(args.gp)} different circuits "
              f"({args.gp}). This is not the recommended 'per-circuit' flow "
              f"— expect weak generalization metrics (see the negative R2 "
              f"discussion).")

    print(f'\n[1/3] Training the degradation model with: {races}')
    try:
        pipeline, metrics = train_degradation_model(races, save_path=model_path)
    except Exception as exc:
        print(f'\nFailed to load/train with FastF1 data: {exc}')
        print('Check your internet connection and the FastF1 cache, then try again.')
        sys.exit(1)

    print(f'Model saved to: {model_path}')
    print('Model metrics:')
    for key, value in metrics.items():
        print(f'  {key}: {value}')

    print(f'\n[2/3] Predicting degradation and suggesting a pit stop window '
          f'(compound={args.compound}, track temp={args.track_temp}°C, '
          f'{args.total_laps} laps)...')
    result = suggest_pit_window(
        pipeline,
        compound=args.compound,
        track_temp=args.track_temp,
        fuel_start_kg=args.fuel_start,
        total_laps=args.total_laps,
        pit_loss_s=args.pit_loss,
    )
    print(f"Suggested optimal pit stop lap: {result['optimal_pit_lap']}")

    # Tie the Monte Carlo risk analysis to THIS specific pit stop: use the
    # degradation model's own predicted curve to estimate how much pace a
    # fresh tyre is worth per lap for this circuit/compound, instead of a
    # generic guessed constant.
    tyre_advantage = estimate_tyre_advantage_s_per_lap(result['curve'])
    print(f"Estimated fresh-tyre pace advantage for {args.compound} at this "
          f"circuit: {tyre_advantage:.3f} s/lap (from the trained degradation model)")

    print(f"\n[3/3] Running the Monte Carlo pit stop risk simulation for "
          f"pitting on lap {result['optimal_pit_lap']} "
          f'({args.n_simulations} simulations per scenario, mean pit loss={args.pit_loss}s, '
          f'SC probability={args.sc_probability})...')
    comparison = compare_scenarios(
        n_simulations=args.n_simulations,
        pit_loss_mean_s=args.pit_loss,
        safety_car_probability_with_sc=args.sc_probability,
        tyre_advantage_s_per_lap=tyre_advantage,
    )
    print(comparison.to_string())
    print(f"\n(Read as: risk of gaining/losing a track position IF you pit on "
          f"lap {result['optimal_pit_lap']}, using this circuit's own fresh-tyre "
          f"pace advantage of {tyre_advantage:.3f} s/lap — not a generic guess.)")

    return {
        'metrics': metrics,
        'model_path': model_path,
        'optimal_pit_lap': result['optimal_pit_lap'],
        'degradation_curve': result['curve'],
        'tyre_advantage_s_per_lap': tyre_advantage,
        'simulation_comparison': comparison,
    }


if __name__ == '__main__':
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)

    args = parse_args()
    run_pipeline(args)