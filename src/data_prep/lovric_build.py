"""Build a 1-minute table from Lovric et al. (2025): reference-grade PM/CO2/VOC/T/RH, low-cost formaldehyde (provider 2), people count."""
import sys, pandas as pd, numpy as np
src, out = sys.argv[1], sys.argv[2]
ref = pd.read_csv(f'{src}/Indoor_referent_device.csv', parse_dates=['Timestamp'])
ref = ref.rename(columns={'Timestamp': 'ts', 'PM2.5 [µg/m³]': 'PM2.5', 'PM10 [µg/m³]': 'PM10', 'CO2 [ppm]': 'CO2', 'VOC [mg/m³]': 'VOC_mgm3', 'T [°C]': 'Temperature', 'rH [%]': 'Humidity'})
ref = ref[['ts', 'PM2.5', 'PM10', 'CO2', 'VOC_mgm3', 'Temperature', 'Humidity']].set_index('ts').resample('1min').mean()
p2 = pd.read_csv(f'{src}/Indoor_sensor_provider_2.csv', parse_dates=['Timestamp']).rename(columns={'Timestamp': 'ts', 'Formaldehyde (µg/m3)': 'HCHO_ugm3'})
p2 = p2[['ts', 'HCHO_ugm3']].set_index('ts').sort_index().resample('1min').mean().ffill(limit=15)
pe = pd.read_csv(f'{src}/People.csv', parse_dates=['Timestamp']).rename(columns={'Timestamp': 'ts', 'People in a room': 'people'})
pe = pe[['ts', 'people']].set_index('ts').sort_index().resample('1min').max().ffill(limit=5)
d = ref.join(p2, how='left').join(pe, how='left').dropna(subset=['PM2.5', 'CO2', 'Temperature', 'Humidity']).reset_index()
d.to_csv(out, index=False)
print(d.ts.min(), d.ts.max(), len(d), 'minutes;', d.ts.dt.date.nunique(), 'days')
print(d.describe().T.round(2))
for k, s in {'CO2>1000': d.CO2 > 1000, 'PM2.5>15': d['PM2.5'] > 15, 'PM10>45': d.PM10 > 45, 'HCHO>100ug/m3 (WHO)': d.HCHO_ugm3 > 100,
             'VOC>0.3 mg/m3 (UBA 1)': d.VOC_mgm3 > 0.3, 'VOC>1 mg/m3 (UBA 3)': d.VOC_mgm3 > 1, 'VOC>3 mg/m3 (UBA 4)': d.VOC_mgm3 > 3}.items():
    print(f'{k:24s} {100*s.mean():5.1f}%')
print('HCHO coverage', d.HCHO_ugm3.notna().mean(), 'people coverage', d.people.notna().mean())
