#!/usr/bin/env python3
"""
tb_group_plot_binned.py - Grouped averages with standard error bands from TensorBoard logs (binned).

Changes vs original:
- Replaced EWMA smoothing with binning on a shared x-grid.
- New args: --bin_count (default 200), --min_bin_count (default 1).
- Each run is binned onto common edges [xmin, xmax] by group; means are computed per bin.
- Group mean and stderr are taken across runs at each bin using only runs that have data in that bin.
- CSV includes x (bin centers), per-group mean and stderr (NaN where insufficient data).
"""
import argparse
import os
import re
import fnmatch
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


# ---------- Utilities ----------
def sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._\-]+", "_", name)


def discover_runs(logdir: str) -> List[str]:
    run_dirs = set()
    for root, _, files in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in files):
            run_dirs.add(root)
    return sorted(run_dirs)


def load_scalars(run_dir: str, tag: str):
    acc = EventAccumulator(run_dir)
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return None, None, None
    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events], dtype=float)
    wall = np.array([e.wall_time for e in events], dtype=float)
    vals = np.array([e.value for e in events], dtype=float)
    return steps, wall, vals


def group_by_patterns(run_dirs: List[str], spec: str) -> Dict[str, List[str]]:
    groups = {}
    entries = [s for s in spec.split(";") if s.strip()]
    for ent in entries:
        if "=" not in ent:
            continue
        name, patterns = ent.split("=", 1)
        name = name.strip()
        pats = [p.strip() for p in patterns.split(",") if p.strip()]
        members = []
        for rd in run_dirs:
            rel = rd.replace("\\", "/")
            for pat in pats:
                if fnmatch.fnmatch(rel, pat):
                    members.append(rd)
                    break
        if members:
            groups[name] = sorted(set(members))
    return groups


def group_by_regex(run_dirs: List[str], regex: str) -> Dict[str, List[str]]:
    rx = re.compile(regex)
    groups = {}
    for rd in run_dirs:
        leaf = os.path.basename(rd.rstrip(os.sep))
        m = rx.match(leaf)
        if not m:
            continue
        key = m.group(1) if m.groups() else m.group(0)
        groups.setdefault(key, []).append(rd)
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k])
    return dict(sorted(groups.items()))


def compute_common_bins(series_list: List[Tuple[np.ndarray, np.ndarray]], bin_count: int):
    xmin = max(np.min(x) for x, _ in series_list)
    xmax = min(np.max(x) for x, _ in series_list)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return None, None
    edges = np.linspace(xmin, xmax, bin_count + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers


def bin_one_series(x: np.ndarray, y: np.ndarray, edges: np.ndarray, min_bin_count: int):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size == 0:
        n = len(edges) - 1
        return np.full(n, np.nan), np.zeros(n, dtype=int)

    idx = np.digitize(x, edges) - 1
    idx = np.clip(idx, 0, len(edges) - 2)

    n_bins = len(edges) - 1
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=int)
    np.add.at(sums, idx, y)
    np.add.at(counts, idx, 1)

    means = np.full(n_bins, np.nan, dtype=float)
    valid = counts >= min_bin_count
    means[valid] = sums[valid] / counts[valid]
    return means, counts


def plot_groups(
    tag_list: List[str],
    groups: Dict[str, List[str]],
    out_dir: str,
    xaxis: str,
    yaxis: str,
    bin_count: int,
    min_bin_count: int,
    title: str,
    single_figure: bool,
):
    os.makedirs(out_dir, exist_ok=True)

    if single_figure:
        plt.figure()
        legend_entries = []

    for tag in tag_list:
        group_curves = {}
        for gname, members in groups.items():
            per_run_xy = []
            for rd in members:
                steps, wall, vals = load_scalars(rd, tag)
                if steps is None:
                    continue
                x = steps if xaxis == "steps" else wall
                y = vals
                per_run_xy.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))

            if len(per_run_xy) < 1:
                continue

            edges, centers = compute_common_bins(per_run_xy, bin_count=bin_count)
            if edges is None:
                continue

            binned_values = []
            valid_mask_list = []
            for x, y in per_run_xy:
                means, counts = bin_one_series(x, y, edges, min_bin_count=min_bin_count)
                binned_values.append(means)
                valid_mask_list.append(counts >= min_bin_count)

            binned_values = np.stack(binned_values, axis=0)
            valid_mask = np.stack(valid_mask_list, axis=0)

            with np.errstate(invalid="ignore"):
                mean = np.nanmean(binned_values, axis=0)

            contrib = np.sum(valid_mask, axis=0)
            se = np.full_like(mean, np.nan, dtype=float)
            multi = contrib >= 2
            if np.any(multi):
                std = np.nanstd(binned_values[:, multi], axis=0, ddof=1)
                se[multi] = std / np.sqrt(contrib[multi])

            group_curves[gname] = (centers, mean, se)

        if not group_curves:
            continue

        if single_figure:
            for gname, (cx, mean, se) in group_curves.items():
                m = np.isfinite(mean)
                plt.plot(cx[m], mean[m], linewidth=1.8)
                m2 = m & np.isfinite(se)
                if np.any(m2):
                    plt.fill_between(cx[m2], mean[m2] - se[m2], mean[m2] + se[m2], alpha=0.25, linewidth=0)
                legend_entries.append(f"{gname} — {tag}")
        else:
            plt.figure()
            for gname, (cx, mean, se) in group_curves.items():
                m = np.isfinite(mean)
                plt.plot(cx[m], mean[m], linewidth=1.8, label=gname)
                m2 = m & np.isfinite(se)
                if np.any(m2):
                    plt.fill_between(cx[m2], mean[m2] - se[m2], mean[m2] + se[m2], alpha=0.25, linewidth=0)
            plt.xlabel(xaxis.upper())
            plt.ylabel(yaxis.upper() if yaxis else "Value")
            plt.title(f"{title}")
            plt.legend(frameon=False)
            plt.tight_layout()
            base = os.path.join(out_dir, sanitize(tag))
            plt.savefig(base + ".png", dpi=300)
            plt.savefig(base + ".pdf")
            plt.close()

            # CSV export
            csv_path = base + ".csv"
            import csv as _csv
            with open(csv_path, "w", newline="") as f:
                writer = _csv.writer(f)
                header = ["x"]
                for gname in group_curves.keys():
                    header += [f"{gname}_mean", f"{gname}_stderr"]
                writer.writerow(header)

                first_key = next(iter(group_curves))
                cx0 = group_curves[first_key][0]
                series_cols = []
                for gname in group_curves.keys():
                    _, mean, se = group_curves[gname]
                    series_cols.append((mean, se))
                for i in range(len(cx0)):
                    row = [cx0[i]]
                    for (mean, se) in series_cols:
                        row += [mean[i], se[i]]
                    writer.writerow(row)

    if single_figure:
        plt.xlabel(xaxis.upper())
        plt.ylabel(yaxis.upper() if yaxis else "Value")
        ttl = f"Overlay: {title}"
        plt.title(ttl)
        plt.legend(legend_entries, frameon=False)
        plt.tight_layout()
        base = os.path.join(out_dir, sanitize("overlay_" + "_".join(tag_list)))
        plt.savefig(base + ".png", dpi=300)
        plt.savefig(base + ".pdf")
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tags", required=True,
        help="Comma-separated scalar tags (e.g., 'eval/return,eval/success_rate')")
    ap.add_argument("--groups", default="", help="Pattern groups, e.g., 'PPO=**/ppo_seed*;SAC=**/sac_seed*'")
    ap.add_argument("--group_by_regex", default="",
        help="Regex that captures a group name from the run leaf dir (group 1). e.g., '^(.*?)-seed\\d+$'")
    ap.add_argument("--xaxis", choices=["steps", "wall_time"], default="steps")
    ap.add_argument("--yaxis", default="", help="y-axis label")

    ap.add_argument("--bin_count", type=int, default=200, help="Number of bins over the common x-range.")
    ap.add_argument("--min_bin_count", type=int, default=1, help="Minimum samples required in a run's bin to include it.")

    # Compatibility (ignored)
    ap.add_argument("--smoothing", type=float, default=0.0, help="(Ignored, binning used)")
    ap.add_argument("--downsample", type=int, default=0, help="(Ignored, binning used)")
    ap.add_argument("--grid_points", type=int, default=600, help="(Ignored)")
    ap.add_argument("--combine_tags", action="store_true",
        help="Plot all tags on the same axis in a single figure.")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--title", default="", help="Optional title for the plot.")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global tag_list
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    run_dirs = discover_runs(args.logdir)
    if args.verbose:
        print(f"Found {len(run_dirs)} runs under {args.logdir}")
    if not run_dirs:
        print("No runs found.")
        return

    if args.groups:
        groups = group_by_patterns(run_dirs, args.groups)
    elif args.group_by_regex:
        groups = group_by_regex(run_dirs, args.group_by_regex)
    else:
        groups = {"all": run_dirs}

    if args.verbose:
        for g, members in groups.items():
            print(f"[Group {g}] {len(members)} runs")

    plot_groups(
        tag_list=tag_list,
        groups=groups,
        out_dir=args.out,
        xaxis=args.xaxis,
        yaxis=args.yaxis,
        bin_count=args.bin_count,
        min_bin_count=args.min_bin_count,
        title=args.title or ", ".join(tag_list),
        single_figure=args.combine_tags,
    )
    print(f"Done. Figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
