"""Average the cleaned 1-min DALTON file to hourly means per sensor (input to Study 3)."""
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src, parse_dates=['ts'], usecols=['ts', 'sensor', 'site', 'site_type', 'T', 'H', 'PMS2_5', 'PMS10', 'CO2', 'VoC'])
h = d.set_index('ts').groupby(['sensor', 'site', 'site_type']).resample('1h')[['T', 'H', 'PMS2_5', 'PMS10', 'CO2', 'VoC']].mean().dropna().reset_index()
h.to_csv(out, index=False); print(len(h), 'sensor-hours')
