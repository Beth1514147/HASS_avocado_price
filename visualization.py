import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Visualize revenue per regions:
def summarize_revenue(df,draw_type):
    df = df[~df.region.isin(['TotalUS','West','SouthCentral','Northeast','Southeast'])]
    fig = plt.figure(figsize=(8,6))
    dfff = df.groupby('region')[draw_type].agg('sum').sort_values(ascending=False)[:10]
    plt.barh(dfff.index, dfff.values)
    plt.title('Top 10 regions have high {}'.format(draw_type))

    return fig,dfff

def revenue_per_region(df,avo_type):
    df_type = df[df.type == avo_type]
    g = sns.FacetGrid(df_type, col = 'region', sharey = False, sharex = False, col_wrap = 3 )
    g.map(sns.lineplot,x = df_type['year_month'], y = df_type['total_volume'])

    return g