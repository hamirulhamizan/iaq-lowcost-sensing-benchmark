"""Reshape the work-from-home workbook (one sheet per variable, one column per site) to long format (input to Study 2)."""
import sys, pandas as pd
src, out = sys.argv[1], sys.argv[2]
x = pd.read_excel(src, sheet_name=None); parts = []
for name, d in x.items():
    var = name.replace('_hourly', '')
    parts.append(d.rename(columns={'timestamp': 'ts'}).melt(id_vars='ts', var_name='site', value_name=var).set_index(['ts', 'site']))
pd.concat(parts, axis=1).reset_index().to_csv(out, index=False); print('written', out)
