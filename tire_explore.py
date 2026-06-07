import fastf1
import pandas as pd

fastf1.Cache.enable_cache('f1_cache')

session = fastf1.get_session(2023, 'Bahrain', 'R')
session.load()

laps = session.laps

ver = laps.pick_driver('VER')

print(ver[['LapNumber', 'LapTime', 'Compound', 'TyreLife']].to_string())