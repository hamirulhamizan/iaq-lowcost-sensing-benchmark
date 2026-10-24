import glob, os, sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
frames = []
for f in sorted(glob.glob(os.path.join(src, '*.csv'))):
    s = os.path.basename(f)[:-4]
    if s.startswith('_') or s == 'F2': continue          # F2 duplicates F1
    d = pd.read_csv(f, parse_dates=['ts'])
    d = d[(d.ts >= '2022-09-01') & (d.ts < '2024-05-01') & (d.valid_frac >= 0.5)]
    frames.append(d)
d = pd.concat(frames, ignore_index=True)
d['site_type'] = d.site.str.extract(r'^([A-Z])')[0].map({'H': 'home', 'A': 'studio apartment', 'R': 'research lab', 'C': 'classroom', 'F': 'canteen'})
d['sensor'] = d.site + '_' + d.file
d.to_csv(out, index=False)
print(len(d), 'minutes;', d.site.nunique(), 'sites;', d.sensor.nunique(), 'sensors;', d.ts.min(), d.ts.max())
print(d.groupby('site_type').agg(sites=('site', 'nunique'), minutes=('ts', 'size')))
