"""Recreate workshop two-panel savings style from completed episode results.
Style source: 6c27ee4:notebooks/visualize_regions.ipynb cells 1,7,9.
Historical CI was normal SE; default here is requested percentile bootstrap.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGIONS = {"us-west-1": "#1f77b4", "us-west-2": "#2a9d8f", "us-east-2": "#9467bd"}
POLICY = "Fixed Start Time Dynamic Parallelism"

def truth(s):
    return s.astype(str).str.lower().isin(["true", "1"])

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=Path("runs/workshop_fixed_start_may2024_results"))
    ap.add_argument("--out", type=Path, default=Path("figures/workshop_figure4_reproduction"))
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-sweep", action="store_true",
                    help="Require all 48 completed 1B--8B model/region workloads")
    ap.add_argument("--ci-method", choices=["bootstrap", "normal"], default="bootstrap",
                    help="Bootstrap as requested, or original workshop mean +/- 1.96 SE")
    args = ap.parse_args()
    if args.resamples < 1:
        raise ValueError("resamples must be positive")
    summary, episodes, sources = [], [], []
    rng = np.random.default_rng(args.seed)
    for path in sorted(args.results.glob("*/*/results.csv")):
        status_path = path.parent / "status.json"
        status = json.loads(status_path.read_text())
        if status.get("status") != "completed" or status.get("returncode") != 0:
            raise ValueError(f"Run not completed: {path}")
        cmd = status["command"]
        model = cmd[cmd.index("--model-name") + 1]
        tokens = int(cmd[cmd.index("--tokens") + 1])
        region = path.parent.name
        if model not in ("gpt2", "llama_1b") or region not in REGIONS:
            continue
        if "--include-job-shift" in cmd:
            raise ValueError(f"Not a fixed-start experiment: {path}")
        df = pd.read_csv(path)
        good = truth(df.completed) & truth(df.feasible)
        keys = ["episode", "offset_h"]
        static = df[good & df.policy.str.startswith("static(")]
        best = static.loc[static.groupby(keys).carbon_g.idxmin(), keys + ["carbon_g"]].rename(columns={"carbon_g": "static_carbon_g"})
        oracle = df[good & (df.policy == POLICY)][keys + ["carbon_g"]]
        paired = oracle.merge(best, on=keys, validate="one_to_one")
        if paired.empty or (paired.static_carbon_g <= 0).any():
            raise ValueError(f"No valid positive-carbon paired reference: {path}")
        values = 100 * (paired.static_carbon_g - paired.carbon_g) / paired.static_carbon_g
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite savings: {path}")
        sample = values.to_numpy()
        means = rng.choice(sample, size=(args.resamples, len(sample)), replace=True).mean(axis=1)
        low, high = np.quantile(means, [0.025, 0.975])
        if args.ci_method == "normal":
            halfwidth = 1.96 * sample.std(ddof=1) / np.sqrt(len(sample)) if len(sample)>1 else 0.0
            low, high = sample.mean()-halfwidth, sample.mean()+halfwidth
        summary.append(dict(model=model, region=region, tokens=tokens, token_b=tokens/1e9,
            mean_pct=sample.mean(), ci_low_pct=low, ci_high_pct=high, paired_n=len(sample),
            total_oracle_n=int((df.policy == POLICY).sum()), feasible_oracle_n=len(oracle),
            dropped_missing_static=len(oracle)-len(paired),
            historical_normal_halfwidth=1.96*sample.std(ddof=1)/np.sqrt(len(sample)) if len(sample)>1 else 0))
        paired = paired.assign(model=model, region=region, tokens=tokens, savings_pct=values)
        episodes.append(paired)
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            status_path=str(status_path), status_sha256=hashlib.sha256(status_path.read_bytes()).hexdigest()))
    table = pd.DataFrame(summary)
    if table.empty or table.duplicated(["model", "region", "tokens"]).any():
        raise ValueError("Empty or duplicate workload results")
    if args.full_sweep:
        expected = {(model, region, n*10**9)
                    for model in ("gpt2", "llama_1b") for region in REGIONS for n in range(1,9)}
        actual = set(zip(table.model, table.region, table.tokens))
        if actual != expected:
            raise ValueError(f"Full sweep incomplete or unexpected budgets: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}")
        if (table.total_oracle_n != 64).any():
            raise ValueError("Original Figure 4 sweep requires 64 oracle episodes per run")
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "figure4_summary.csv", index=False)
    pd.concat(episodes, ignore_index=True).to_csv(args.out / "figure4_episode_savings.csv", index=False)
    plt.rcdefaults()
    plt.rcParams.update({"figure.dpi":150, "savefig.dpi":300,
        "font.family":"DejaVu Sans", "font.size":8,
        "axes.spines.top":False, "axes.spines.right":False,
        "axes.linewidth":0.6, "axes.labelsize":9,
        "xtick.labelsize":8, "ytick.labelsize":8,
        "xtick.major.width":0.6, "ytick.major.width":0.6,
        "xtick.major.size":3, "ytick.major.size":3,
        "pdf.fonttype":42, "ps.fonttype":42})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.65), sharey=True)
    for ax, model, title in zip(axes, ["gpt2", "llama_1b"], ["GPT-2", "Llama 1B"]):
        sub = table[table.model == model]
        if sub.empty:
            raise ValueError(f"Missing model {model}")
        for region, color in REGIONS.items():
            rs = sub[sub.region == region].sort_values("tokens")
            if rs.empty:
                raise ValueError(f"Missing {model}/{region}")
            y = rs.mean_pct.to_numpy()
            ax.plot(rs.token_b, y, marker="o", markersize=3,
                    markeredgewidth=0.4, linewidth=1.0,
                    linestyle="-" if len(rs) > 1 else "none",
                    color=color, label=region, zorder=3)
            # Percentile bounds are drawn directly without changing the mean.
            for x, low, high in zip(rs.token_b, rs.ci_low_pct, rs.ci_high_pct):
                ax.errorbar([x], [(low+high)/2],
                            yerr=[[(high-low)/2], [(high-low)/2]],
                            fmt="none", color=color, capsize=2,
                            elinewidth=0.65, capthick=0.65, zorder=2)
        ax.axhline(0, color="#555555", linewidth=0.6, zorder=1)
        ax.set_title(title, fontsize=10, pad=5)
        ax.set_xlabel("Training tokens (billions)", labelpad=4)
        ticks = sorted(sub.token_b.unique())
        ax.set_xticks(ticks, [f"{x:g}" for x in ticks])
        ax.grid(True, axis="both", color="#d9d9d9", linewidth=0.45, alpha=0.6)
        ax.set_axisbelow(True)
        ax.margins(x=0.06)
        if len(ticks) == 1:
            ax.set_xlim(ticks[0]-0.5, ticks[0]+0.5)
    # Include actual confidence bounds and zero, with modest symmetric padding.
    lower = min(0.0, float(table.ci_low_pct.min()), float(table.mean_pct.min()))
    upper = max(0.0, float(table.ci_high_pct.max()), float(table.mean_pct.max()))
    padding = 0.06 * (upper-lower) if upper > lower else 0.01
    axes[0].set_ylim(lower-padding, upper+padding)
    axes[0].set_ylabel("Carbon saved (%)", labelpad=5)
    axes[-1].legend(frameon=False, loc="upper right", fontsize=8,
                    handlelength=1.8, handletextpad=0.5, labelspacing=0.35,
                    borderaxespad=0.3)
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.20, top=0.89, wspace=0.12)
    for extension in ["png", "pdf"]:
        fig.savefig(args.out / f"figure4_reproduction.{extension}", dpi=300)
    plt.close(fig)
    (args.out / "provenance.json").write_text(json.dumps(dict(
        style_source="6c27ee4:notebooks/visualize_regions.ipynb cells 1,7,9",
        bootstrap_resamples=args.resamples, bootstrap_seed=args.seed,
        method=("Per-workload iid episode percentile bootstrap of mean paired savings; 2.5/97.5 percentiles"
                if args.ci_method == "bootstrap" else "Original workshop normal interval: mean +/- 1.96 sample SE"),
        caveats=["Historical notebook used 1.96 SE, not bootstrap; normal halfwidth exported for comparison",
                 "Episodes may overlap in historical time; CI is conditional episode-resampling uncertainty, not measurement uncertainty",
                 "Only completed measured budgets are plotted; single-budget series have no connecting segments"],
        sources=sources), indent=2)+"\n")
    print(table.to_string(index=False))
    print(f"Saved figures and source statistics to {args.out}")

if __name__ == "__main__":
    main()
