import numpy as np
import pandas as pd


def _sample_gaps(rng, n, scenario):
    """Distribution of the gap (s) to the car ahead/behind on track.

    'close_battle': direct fight, short gaps (DRS-zone-like).
    'clear_air'   : no immediate fight, larger and more spread-out gaps.
    """
    if scenario == 'close_battle':
        gap_ahead = np.clip(rng.exponential(1.5, n), 0, 5)
        gap_behind = np.clip(rng.exponential(1.5, n), 0, 5)
    elif scenario == 'clear_air':
        gap_ahead = np.clip(rng.normal(8.0, 4.0, n), 0, None)
        gap_behind = np.clip(rng.normal(8.0, 4.0, n), 0, None)
    else:
        raise ValueError("scenario must be 'close_battle' or 'clear_air'")
    return gap_ahead, gap_behind


def _sample_pit_loss(rng, n, is_safety_car, pit_loss_mean_s, pit_loss_std_s,
                      safety_car_pit_loss_s):
    normal = rng.normal(pit_loss_mean_s, pit_loss_std_s, n)
    sc = rng.normal(safety_car_pit_loss_s, pit_loss_std_s * 0.5, n)
    return np.clip(np.where(is_safety_car, sc, normal), 0, None)


def simulate_pit_stop_risk(
    n_simulations=20_000,
    scenario='close_battle',
    pit_loss_mean_s=22.0,
    pit_loss_std_s=1.5,
    safety_car_probability=0.0,
    safety_car_pit_loss_s=12.0,
    rival_reaction_same_lap_prob=0.6,
    rival_reaction_delay_mean_laps=2.0,
    tyre_advantage_s_per_lap=0.15,
    undercut_horizon_laps=5,
    rng_seed=42,
):
    rng = np.random.default_rng(rng_seed)
    n = n_simulations

    gap_ahead, gap_behind = _sample_gaps(rng, n, scenario)

    is_sc = rng.random(n) < safety_car_probability
    our_pit_loss = _sample_pit_loss(
        rng, n, is_sc, pit_loss_mean_s, pit_loss_std_s, safety_car_pit_loss_s
    )

    def _duel(gap, is_ahead):
        reacts_same_lap = rng.random(n) < rival_reaction_same_lap_prob
        # Only gets the SC benefit if they stop on the same lap as us.
        rival_is_sc = is_sc & reacts_same_lap
        rival_pit_loss = _sample_pit_loss(
            rng, n, rival_is_sc, pit_loss_mean_s, pit_loss_std_s, safety_car_pit_loss_s
        )
        delay = np.where(
            reacts_same_lap, 0,
            rng.geometric(1.0 / rival_reaction_delay_mean_laps, n)
        )
        delay = np.clip(delay, 0, undercut_horizon_laps)

        pace_gain = tyre_advantage_s_per_lap * delay

        if is_ahead:
            final_gap = gap + our_pit_loss - pace_gain - rival_pit_loss
            return final_gap < 0  # we passed them (undercut paid off)
        else:
            final_gap = gap - our_pit_loss + pace_gain + rival_pit_loss
            return final_gap < 0  # they passed us

    gained_position_ahead = _duel(gap_ahead, is_ahead=True)
    lost_position_behind = _duel(gap_behind, is_ahead=False)

    df = pd.DataFrame({
        'safety_car': is_sc,
        'our_pit_loss_s': our_pit_loss,
        'gap_ahead_s': gap_ahead,
        'gap_behind_s': gap_behind,
        'gained_position_from_ahead': gained_position_ahead,
        'lost_position_to_behind': lost_position_behind,
    })

    summary = {
        'p_gain_position_on_car_ahead': float(df['gained_position_from_ahead'].mean()),
        'p_lose_position_to_car_behind': float(df['lost_position_to_behind'].mean()),
        'mean_pit_loss_s': float(df['our_pit_loss_s'].mean()),
        'safety_car_probability_used': safety_car_probability,
        'scenario': scenario,
        'n_simulations': n,
    }
    return summary, df


def compare_scenarios(safety_car_probability_with_sc=0.3, **kwargs):
    """Runs the simulation with and without Safety Car, for both traffic
    scenarios (close battle / clear air), and returns everything in a
    DataFrame for comparison."""
    rows = {}
    for scenario in ('close_battle', 'clear_air'):
        base_kwargs = dict(kwargs)
        base_kwargs['scenario'] = scenario

        no_sc_kwargs = dict(base_kwargs)
        no_sc_kwargs['safety_car_probability'] = 0.0
        summary_no_sc, _ = simulate_pit_stop_risk(**no_sc_kwargs)
        rows[f'{scenario}_no_SC'] = summary_no_sc

        sc_kwargs = dict(base_kwargs)
        sc_kwargs['safety_car_probability'] = safety_car_probability_with_sc
        summary_sc, _ = simulate_pit_stop_risk(**sc_kwargs)
        rows[f'{scenario}_with_SC'] = summary_sc

    return pd.DataFrame(rows).T


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 160)
    comparison = compare_scenarios(
        n_simulations=20_000,
        pit_loss_mean_s=22.0,
        safety_car_probability_with_sc=0.3,
    )
    print(comparison)