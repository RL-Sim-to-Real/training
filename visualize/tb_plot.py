#!/usr/bin/env python3
"""
tb_group_plot.py — Grouped averages with standard error bands from TensorBoard logs.

Features
- Discover runs recursively under --logdir
- Define groups:
    (A) Explicit patterns: --groups "PPO=**/PPO_seed*;SAC=**/SAC_seed*"
    (B) Regex capture from run leaf dir: --group_by_regex "^(.*?)(?:_seed\\d+)?$"
- For each tag, align runs onto a common x-grid of STEPS or WALL_TIME
  (linear interpolation), then compute group mean and standard error
- Plot mean ± stderr as a shaded band
- Optionally overlay multiple tags on the same axis via --combine_tags
- Smoothing (EWMA) and uniform downsampling for lighter figures
- Save per-figure PNG + PDF plus CSV with grid/means/stderr

Examples
  python tb_group_plot.py --logdir ./logs --out ./figs \
      --groups "PPO=**/ppo_seed*;SAC=**/sac_seed*" \
      --tags "eval/return" --smoothing 0.9 --downsample 400

  python tb_group_plot.py --logdir ./logs --out ./figs \
      --group_by_regex "^(.*?)-seed\\d+$" \
      --tags "eval/return,eval/success_rate" --combine_tags
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
def ewma(values, alpha: float):
    if alpha <= 0:
        return np.asarray(values, dtype=float)
    out = np.empty(len(values), dtype=float)
    m = None
    for i, v in enumerate(values):
        m = v if m is None else alpha * m + (1 - alpha) * v
        out[i] = m
    return out


def uniform_downsample(x, y, n_points: int):
    if n_points is None or n_points <= 0 or len(x) <= n_points:
        return x, y
    idxs = np.linspace(0, len(x) - 1, num=n_points).astype(int)
    idxs = np.unique(idxs)
    return x[idxs], y[idxs]


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
    """
    spec example: "PPO=**/PPO_seed*;SAC=**/SAC_seed*"
    Supports multiple ; separated groups; each RHS can be comma-separated patterns.
    """
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
    # sort for stability
    for k in list(groups.keys()):
        groups[k] = sorted(groups[k])
    return dict(sorted(groups.items()))


def make_common_grid(series_list: List[Tuple[np.ndarray, np.ndarray]], grid_points: int):
    """
    series_list: list of (x, y) arrays for runs in a group.
    Returns grid_x, aligned_y [n_runs, n_grid]
    Aligns by intersection of x-range across runs to avoid extrapolation artifacts.
    """
    # Determine overlap
    xmin = max(np.min(x) for x, _ in series_list)
    xmax = min(np.max(x) for x, _ in series_list)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return None, None
    grid_x = np.linspace(xmin, xmax, grid_points)
    aligned = []
    for x, y in series_list:
        # Remove duplicates in x to allow interpolation
        uniq_idx = np.unique(x, return_index=True)[1]
        x_u = x[np.sort(uniq_idx)]
        y_u = y[np.sort(uniq_idx)]
        # Interpolate
        y_interp = np.interp(grid_x, x_u, y_u)
        aligned.append(y_interp)
    aligned = np.stack(aligned, axis=0)  # [n_runs, n_grid]
    return grid_x, aligned


def plot_groups(
    tag_list: List[str],
    groups: Dict[str, List[str]],
    out_dir: str,
    xaxis: str,
    yaxis: str,
    smoothing: float,
    downsample: int,
    grid_points: int,
    title: str,
    single_figure: bool,
):
    os.makedirs(out_dir, exist_ok=True)

    if single_figure:
        # One figure overlaying multiple tags (and all groups)
        plt.figure()
        legend_entries = []

    for tag in tag_list:
        # Collect per-group aligned data
        group_curves = {}  # name -> (grid_x, mean, stderr)
        for gname, members in groups.items():
            per_run_xy = []
            for rd in members:
                steps, wall, vals = load_scalars(rd, tag)
                if steps is None:
                    continue
                x = steps if xaxis == "steps" else wall
                y = vals
                if smoothing > 0:
                    y = ewma(y, smoothing)
                if downsample and downsample > 0:
                    x, y = uniform_downsample(np.asarray(x), np.asarray(y), downsample)
                per_run_xy.append((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
            if len(per_run_xy) < 1:
                continue
            grid_x, aligned = make_common_grid(per_run_xy, grid_points)
            if grid_x is None:
                continue
            mean = aligned.mean(axis=0)
            stderr = (
                aligned.std(axis=0, ddof=1) / np.sqrt(aligned.shape[0])
                if aligned.shape[0] > 1
                else np.zeros_like(mean)
            )
            group_curves[gname] = (grid_x, mean, stderr)

        if not group_curves:
            continue

        # Plotting
        if single_figure:
            for gname, (gx, mean, se) in group_curves.items():
                plt.plot(gx, mean, linewidth=1.8)
                plt.fill_between(gx, mean - se, mean + se, alpha=0.25, linewidth=0)
                legend_entries.append(f"{gname} — {tag}")
        else:
            plt.figure()
            for gname, (gx, mean, se) in group_curves.items():
                plt.plot(gx, mean, linewidth=1.8, label=gname)
                plt.fill_between(gx, mean - se, mean + se, alpha=0.25, linewidth=0)
            plt.xlabel(xaxis.upper())
            plt.ylabel(yaxis.upper())
            plt.title(f"{title}")
            plt.legend(frameon=False)
            plt.tight_layout()
            base = os.path.join(out_dir, sanitize(tag))
            plt.savefig(base + ".png", dpi=300)
            plt.savefig(base + ".pdf")
            plt.close()

            # Write CSV (grid, and per-group mean/stderr)
            csv_path = base + ".csv"
            import csv as _csv

            with open(csv_path, "w", newline="") as f:
                writer = _csv.writer(f)
                header = ["x"]
                for gname in group_curves.keys():
                    header += [f"{gname}_mean", f"{gname}_stderr"]
                writer.writerow(header)
                # assume equal grid for all groups (we aligned the same way per tag)
                first_key = next(iter(group_curves))
                gx0 = group_curves[first_key][0]
                # Make a dict for fast lookup
                series_cols = []
                for gname in group_curves.keys():
                    gx, mean, se = group_curves[gname]
                    series_cols.append((mean, se))
                for i in range(len(gx0)):
                    row = [gx0[i]]
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
    ap.add_argument(
        "--tags",
        required=True,
        help="Comma-separated scalar tags (e.g., 'eval/return,eval/success_rate')",
    )
    ap.add_argument(
        "--groups",
        default="",
        help="Pattern groups, e.g., 'PPO=**/ppo_seed*;SAC=**/sac_seed*'",
    )
    ap.add_argument(
        "--group_by_regex",
        default="",
        help="Regex that captures a group name from the run leaf dir (group 1). e.g., '^(.*?)-seed\\d+$'",
    )
    ap.add_argument("--xaxis", choices=["steps", "wall_time"], default="steps")
    ap.add_argument("--yaxis", default="", help="y-axis label")
    ap.add_argument("--smoothing", type=float, default=0.0)
    ap.add_argument("--downsample", type=int, default=0)
    ap.add_argument(
        "--grid_points", type=int, default=600, help="Interpolation grid points for averaging."
    )
    ap.add_argument(
        "--combine_tags",
        action="store_true",
        help="Plot all tags on the same axis in a single figure.",
    )
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--title", default="",
                       help="Optional title for the plot. If not provided, uses tag names.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tag_list = [t.strip() for t in args.tags.split(",") if t.strip()]
    run_dirs = discover_runs(args.logdir)
    if args.verbose:
        print(f"Found {len(run_dirs)} runs under {args.logdir}")

    if not run_dirs:
        print("No runs found.")
        return

    # Build groups
    if args.groups:
        groups = group_by_patterns(run_dirs, args.groups)
    elif args.group_by_regex:
        groups = group_by_regex(run_dirs, args.group_by_regex)
    else:
        # Default: single group named "all"
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
        smoothing=args.smoothing,
        downsample=args.downsample,
        grid_points=args.grid_points,
        title=args.title or ", ".join(tag_list),
        single_figure=args.combine_tags,
    )
    print(f"Done. Figures saved to: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
