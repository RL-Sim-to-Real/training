
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
axis_tag_names = {
    "episode/success": "Success Rate",
    "episode/sum_reward": "Episodic Return",
    "episode/floor_collision": "Floor Collisions",
    "episode/jerk": "Jerk",
}

tb_tags_list = [
    "episode/success",
    "episode/sum_reward",
    "episode/floor_collision",
    "episode/jerk"
]


def plot_group_results(run_dfs, tags, sub_plot, rg_names:list[str], color="", bin_size=20):
    fig, axes = sub_plot
    sns.set_style("white")
    tags = tb_tags_list  # ['episode/success', 'episode/sum_reward', 'episode/floor_collision', 'episode/jerk']

    # choose one distinct color per tag, use same color for all seeds within that tag
    
    action_color = color  # use first color for this action space
    for (ax, tag) in zip(axes, tags):
        df_tag = run_dfs.get(tag)


        df_tag = df_tag.sort_index()
        n = len(df_tag)
        print(f"Processing tag '{tag}' with {n} rows")
        if n <= bin_size:
            binned_df = df_tag.copy()
            rep_steps = df_tag.index.to_numpy()
        else:
            group_idx = np.arange(n) // bin_size
            binned_df = df_tag.groupby(group_idx).mean()
            steps_arr = df_tag.index.to_numpy()
            groups = np.unique(group_idx)
            rep_steps = np.array([steps_arr[g * bin_size] if g * bin_size < n else steps_arr[-1] for g in groups])

        final_vals = binned_df.apply(lambda col: col.dropna().iloc[-1] if col.dropna().size else np.nan)
        print(final_vals)
        med = final_vals.median()
        med_col = final_vals.subtract(med).abs().idxmin()    

        # plot every seed using the same color but lighter (lower alpha) and thin line
        # if this tag is 'episode/jerk' use a log y-axis; ensure values > 0 before switching
        if tag == "episode/jerk":
            # replace non-positive values with a small positive epsilon so log scale works
            if (binned_df <= 0).any().any():
                pos_vals = binned_df[binned_df > 0].stack()
                min_pos = pos_vals.min() if not pos_vals.empty else 1e-8
                eps = max(min_pos * 1e-3, 1e-12)
                binned_df = binned_df.clip(lower=eps)
            ax.set_yscale("log")
        for col in binned_df.columns:
            ax.plot(rep_steps, binned_df[col].to_numpy(), color=action_color, alpha=0.25, linewidth=1)

        # highlight the seed closest to median final value with same color but thicker/opaque
        if med_col in binned_df.columns:
            ax.plot(rep_steps, binned_df[med_col].to_numpy(), color=action_color, linewidth=1, alpha=1.0, label=f"{med_col}")

        ax.set_xlabel("Env Step")
        ax.set_ylabel(axis_tag_names.get(tag, tag))

    return fig



def get_run_group_dfs(group_name:str="", run_groups: dict = {}):
    runs = run_groups.get(group_name, [])

    dfs = {}
    for tag in tb_tags_list:
        series_list = []
        for ef in runs:
            ea = EventAccumulator(ef)
            ea.Reload()
            scalars_tags = ea.Tags().get("scalars", [])
            if tag not in scalars_tags:
                continue
            scalars = ea.Scalars(tag)
            if not scalars:
                continue
            steps = [int(s.step) for s in scalars]
            vals = [s.value for s in scalars]
            parent = os.path.basename(os.path.dirname(ef))
            s = pd.Series(data=vals, index=steps, name=parent)
            series_list.append(s)
        if series_list:
            df = pd.concat(series_list, axis=1).sort_index()
            dfs[tag] = df

    # dfs now maps each tag to a DataFrame (index=step, columns=runs)
    return dfs



def plot_results(run_groups, legend_names):
    # use Times New Roman and increase tick/axis title sizes
    print("generating plot")
    sns.set_style("white")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif", "Liberation Serif", "Times New Roman"]
    plt.rcParams["xtick.labelsize"] = 20
    plt.rcParams["ytick.labelsize"] = 20
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.titlesize"] = 20
    plt.rcParams["legend.fontsize"] = 20
    

    tags = tb_tags_list
    fig, axes = plt.subplots(1, len(tags), figsize=(30, 5), constrained_layout=True)
    rg_names = run_groups.keys()

    
    
    palette = sns.color_palette("tab10", n_colors=len(tags))
    for indx,g_name in tqdm(enumerate(rg_names)):
        dfs = get_run_group_dfs(g_name, run_groups=run_groups)
        fig = plot_group_results(dfs, tags, rg_names=rg_names, color=palette[indx], sub_plot=(fig, axes))
    # display the final combined figure and save to disk

    for indx, ax in enumerate(axes):
        labels = [legend_names[n] for n in list(rg_names)]
        handles = [ax.plot([], [], color=palette[i], lw=2)[0] for i in range(len(labels))]
        ax.legend(handles, labels)


    
    fig.savefig("all_results.pdf", bbox_inches="tight")
