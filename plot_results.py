import pandas as pd
import hvplot.pandas

df_all = pd.read_csv('./all_results.csv')
df_all['Exp'] = df_all['Experiment'].str.slice(stop=-2)
df_win = pd.read_csv('./all_results_win.csv')
df_win['Exp'] = df_win['Experiment'].str.slice(stop=-2)

p = df_all.loc[df_all['Exp'].str.contains('acl18')].hvplot.scatter(
    x='Test Type', y='Test Accuracy', c='Exp', height=400, title='Results ACL18'
).opts(legend_opts={'location': (5, 100)})

p = df_all.loc[df_all['Exp'].str.contains('kdd17')].hvplot.scatter(
    x='Test Type', y='Test Accuracy', c='Exp', height=400, title='Results KDD17'
).opts(legend_opts={'location': (5, 100)})

df_win['Test Win Type'] = df_win['Test Type'] + '-' + df_win['Window Size'].astype(str)
p = df_win.loc[df_win['Exp'].str.contains('acl18')].hvplot.scatter(
    x='Test Win Type', y='Test Accuracy', c='Exp', height=400, width=1000, title='Results ACL18 by windows'
).opts(legend_opts={'location': (5, 100)})

p = df_win.loc[df_win['Exp'].str.contains('kdd17')].hvplot.scatter(
    x='Test Win Type', y='Test Accuracy', c='Exp', height=400, width=1000, title='Results KDD17 by windows'
).opts(legend_opts={'location': (5, 100)})
