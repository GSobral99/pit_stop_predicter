import fastf1
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')


def load_race(year, gp, session_type='R', verify_contains=None):
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(year, gp, session_type)
    session.load()

    event_name = str(session.event.get('EventName', ''))
    location = str(session.event.get('Location', ''))
    print(f"[data_loader] gp='{gp}' -> resolved to '{event_name}' "
          f"({location}), {year}, session '{session_type}'")

    if verify_contains is not None:
        haystack = f'{event_name} {location}'.lower()
        if verify_contains.lower() not in haystack:
            raise ValueError(
                f"The resolved race ('{event_name}', '{location}') does not "
                f"contain '{verify_contains}'. Double-check the gp argument "
                f"passed (gp='{gp}') - it may have been resolved incorrectly."
            )

    return session

if __name__ == '__main__':
    session = load_race(2023, 'Monza', 'R')
    print(session.laps.head())