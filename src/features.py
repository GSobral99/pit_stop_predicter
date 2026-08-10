from src.data_loader import load_race

def clean_laps(laps):
    """Aplica os filtros de qualidade às voltas."""
    return laps[
        (laps['IsAccurate'] == True) &
        (laps['PitInTime'].isna()) &
        (laps['PitOutTime'].isna()) &
        (laps['TrackStatus'] == '1')
    ].copy()

def build_features(year, gp, session_type='R'):
    session = load_race(year, gp, session_type)
    laps = clean_laps(session.laps)
    
    # TODO: juntar temperatura da pista
    # TODO: estimar fuel load
    # TODO: converter LapTime para segundos
    
    return laps