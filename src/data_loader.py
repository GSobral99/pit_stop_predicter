import fastf1
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(BASE_DIR, '..', 'data', 'raw')

def load_race(year, gp, session_type='R'):
    fastf1.Cache.enable_cache(CACHE_DIR)
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    return session

if __name__ == '__main__':
    session = load_race(2023, 'Monza', 'R')
    print(session.laps.head())