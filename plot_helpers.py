import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd


my_template = go.layout.Template(
    layout=dict(
        font_color='#000000',
        #font_family="Open Sans",
        plot_bgcolor = "#ffffff",  #rgba(212,218,220,255)
        paper_bgcolor = "#ffffff", 
        legend=dict(bgcolor='rgba(0,0,0,0)'), #"#ffffff"
        title={'y': 1.0, 'x': .06},
        font_size=13,
        uniformtext_minsize=11, 
        uniformtext_mode='hide',
        margin=dict(l=15, r=0, t=35, b=0),
))

pio.renderers.default = 'plotly_mimetype+notebook_connected'
pio.templates["my"] = my_template
pio.templates.default = "simple_white+xgridoff+ygridoff+my"
#pio.kaleido.scope.mathjax= None

def init_stats_dict(network_files, keys, name):
    
    stats_dict = {
        key: pd.concat([pd.DataFrame(index=network_files.index)]) #, keys=[key], names=[name]
        for key in keys}

    return stats_dict

def drop_index_levels(df, to_drop=[]):
    """
    Drop index levels from the dataframe.
    """
    df = df.copy()

    for level in to_drop:
        if level in df.index.names and df.index.get_level_values(level).nunique() == 1:
            print(f"Dropping index level {level} with only one unique value: {df.index.get_level_values(level).unique()[0]}")
            df = df.droplevel(level)
        elif level in df.index.names:
            print(f"Index level {level} has multiple unique values and should be maintained: {df.index.get_level_values(level).unique()}")

    return df

def set_scen_col(df):
    """
    Set combined scenario str (for plots).
    This is specific to the experiment and should be adapted for other experiments.
    """
    
    df = df.copy()
    
    df["scen"] = df["opts"].astype(str) + "-" + df["demand"].astype(str)
    df = df.drop(columns=["h2export", "opts", "demand"])

    return df

def update_layout(fig):
    fig.update_traces(textposition='inside', textangle=0)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    return

def prepare_dataframe(stats_df, idx_group):
    df = stats_df.copy().loc[idx_group].reset_index()
    df = set_scen_col(df)
    df = df.melt(id_vars=["run_name", "scen", "year", "country"]).groupby(
        ["run_name", "scen", "year", "country", "variable"], as_index=False
    ).sum()


    return df

def get_supply_demand_from_balance(stats_df, threshold=0.01, round=0):
    """
    Get supply and demand from balance.
    """
    supply_df = stats_df[stats_df["value"]>=threshold].copy()
    supply_df = supply_df.groupby(["scen","year","variable"], as_index=False).sum()
    supply_sum_df = supply_df.groupby(["scen","year"]).sum(numeric_only=True).round(round)

    demand_df = stats_df[stats_df["value"]<=-threshold].copy()
    demand_df["value"] *= -1 
    demand_sum_df = demand_df.groupby(["scen","year"]).sum(numeric_only=True).round(round)

    return supply_df, supply_sum_df, demand_df, demand_sum_df

def save_plotly_fig(df, fig, output_dir, fig_name):
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir/f"{fig_name}.csv", index=False)
    fig.write_image(output_dir/f"{fig_name}.svg", engine="kaleido")