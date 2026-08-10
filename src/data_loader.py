import fastf1

fastf1.Cache.enable_cache('data/raw')

session = fastf1.get_session(2023, 'Monza', 'R')
session.load()

print(session.laps.head())