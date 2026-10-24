"""Download DALTON processed files via GitHub LFS media host, verify sha256 against the LFS pointer,
drop zero-filled gap rows, and aggregate to 1-minute means. Output: one CSV per site in OUT."""
import os, re, sys, io, hashlib, urllib.request, concurrent.futures as cf
import pandas as pd
REPO, OUT = sys.argv[1], sys.argv[2]
BASE = 'https://media.githubusercontent.com/media/prasenjit52282/dalton-dataset/main/'
NUM = ['T', 'H', 'PMS1', 'PMS2_5', 'PMS10', 'CO2', 'NO2', 'CO', 'VoC', 'C2H5OH']
jobs = []
for root, _, fs in os.walk(os.path.join(REPO, 'Processed')):
    for f in fs:
        if f.endswith('.csv'):
            p = os.path.join(root, f); s = open(p).read()
            oid = re.search(r'oid sha256:(\w+)', s).group(1)
            jobs.append((os.path.relpath(p, REPO), oid))
jobs.sort()
done_log = os.path.join(OUT, '_done.txt')
done = set(open(done_log).read().split()) if os.path.exists(done_log) else set()
def work(job):
    rel, oid = job
    if rel in done: return rel, 'skip', 0
    for attempt in range(3):
        try:
            data = urllib.request.urlopen(BASE + rel, timeout=300).read()
            if hashlib.sha256(data).hexdigest() != oid: raise ValueError('checksum mismatch')
            d = pd.read_csv(io.BytesIO(data), low_memory=False)
            d['ts'] = pd.to_datetime(d['ts'])
            raw_n = len(d)
            d = d[~((d['T'] == 0) & (d['H'] == 0) & (d['CO2'] == 0))]
            if d.empty: return rel, 'empty', raw_n
            g = d.set_index('ts')
            m = g[NUM].resample('1min').mean()
            m['n_sec'] = g['T'].resample('1min').count()
            m['valid_frac'] = g['Valid'].resample('1min').mean()
            m['valid_co2_frac'] = g['Valid_CO2'].resample('1min').mean()
            m = m[m.n_sec > 0].reset_index()
            parts = rel.split('/')
            m['site'] = parts[1]; m['file'] = parts[-1].replace('.csv', ''); m['date'] = parts[2]
            return rel, m, raw_n
        except Exception as e:
            err = e
    return rel, f'error {err}', 0
results = {}
with cf.ThreadPoolExecutor(4) as ex:
    for i, (rel, m, n) in enumerate(ex.map(work, jobs), 1):
        site = rel.split('/')[1]
        if isinstance(m, pd.DataFrame):
            path = os.path.join(OUT, f'{site}.csv')
            m.to_csv(path, mode='a', header=not os.path.exists(path), index=False)
        status = 'ok' if isinstance(m, pd.DataFrame) else m
        with open(done_log, 'a') as fh:
            if status in ('ok', 'empty', 'skip'): fh.write(rel + '\n')
        print(f'{i}/{len(jobs)} {rel} {status} {n}', flush=True)
