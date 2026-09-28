"""Regenerate preprint figures from compact frozen artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "figure_inputs"
FIG = ROOT / "figures"

RED = "#c9252d"
LIGHT_RED = "#f3a6a2"
GRAY = "#777777"
LIGHT_GRAY = "#cfcfcf"
BLUE = "#2b6cb0"
GREEN = "#2b8a3e"
MAP_EMPTY = "#1a1a1a"
MAP_EMPTY_EDGE = "#555555"
MAP_EXISTING_LAB = "#2c7fb8"
MAP_LAB = "#20c95a"
MAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "magma_lifted", plt.get_cmap("magma")(np.linspace(0.32, 1.0, 256))
)


def _save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _q(vals, probs=(5, 25, 50, 75, 95)) -> dict[str, float]:
    return {f"p{p}": float(np.percentile(vals, p)) for p in probs}


def _kfmt(x, _pos=None) -> str:
    if x >= 10_000:
        return f"{x / 1000:,.0f}k"
    if x >= 1_000:
        return f"{x / 1000:,.1f}k"
    return f"{x:,.0f}"


def fig_response_functions() -> None:
    post = np.load(DATA / "sq_draws.npz")
    deaths = np.linspace(0, 260, 400)
    hill_n = 8.7
    brakes = 1.0 / (1.0 + (deaths[None, :] / post["cum_half"][:, None]) ** hill_n)
    r0 = post["R0"][:, None]
    r_floor = post["R_floor"][:, None]
    local_scale = (r_floor + (r0 - r_floor) * brakes) / r0
    q05, q25, q50, q75, q95 = np.percentile(local_scale, [5, 25, 50, 75, 95], axis=0)

    days = np.linspace(0, 180, 400)
    nat = 0.25 + 0.75 / (1.0 + (days / 90.0) ** 2.0)

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.15))
    ax[0].fill_between(deaths, q05, q95, color=LIGHT_RED, alpha=0.30, lw=0, label="90% CrI")
    ax[0].fill_between(deaths, q25, q75, color=LIGHT_RED, alpha=0.55, lw=0, label="50% CrI")
    ax[0].plot(deaths, q50, color=RED, lw=2.0, label="Median")
    ax[0].set_xlabel("Lagged local visible deaths")
    ax[0].set_ylabel(r"Local scale, $S_z(t)$")
    ax[0].set_ylim(0, 1.03)
    ax[0].set_title("A. Local awareness")
    ax[0].legend(loc="lower left", frameon=False, fontsize=8, handlelength=1.6)

    ax[1].plot(days, nat, color=BLUE, lw=2.0)
    ax[1].axvline(90, color="0.35", ls=":", lw=1.0)
    ax[1].axhline(0.25, color="0.35", ls="--", lw=1.0)
    ax[1].set_xlabel("Days after 2 Aug 2026")
    ax[1].set_ylabel(r"National scale, $S_{\mathrm{nat}}(t)$")
    ax[1].set_ylim(0, 1.03)
    ax[1].set_title("B. Assumed national ramp")
    ax[1].text(
        0.05,
        0.09,
        r"$T_{\mathrm{nat}}=90$ d" "\n" r"$S_{\min}=0.25$",
        transform=ax[1].transAxes,
        fontsize=8,
        color="0.35",
    )

    for a in ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "fig02_response_functions.png")


def fig_status_quo() -> None:
    from matplotlib.ticker import FuncFormatter

    obs = json.loads((DATA / "observed_at_cutoff.json").read_text())
    base = np.genfromtxt(DATA / "status_quo_540.csv", delimiter=",", names=True)
    confirmed_cases = base["c_540"]
    latent_cases = base["n_540"]
    confirmed_deaths = base["cd_540"]
    latent_deaths = base["d_540"]

    wa_cases_confirmed = 15_250
    wa_cases_reported = 28_616
    wa_deaths_reported = 11_310
    panels = [
        (
            "A. Cases",
            "Cumulative cases at +540 d",
            confirmed_cases,
            latent_cases,
            obs["confirmed_cases"],
            [
                (wa_cases_confirmed, "2014 West Africa\nlab-confirmed", "-."),
                (wa_cases_reported, "2014 West Africa\nreported", ":"),
            ],
        ),
        (
            "B. Deaths",
            "Cumulative deaths at +540 d",
            confirmed_deaths,
            latent_deaths,
            obs["confirmed_deaths"],
            [(wa_deaths_reported, "2014 West Africa\nreported", ":")],
        ),
    ]

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.15))
    for a, (title, xlabel, confirmed, latent, today, anchors) in zip(ax, panels):
        hi = max(np.percentile(latent, 99), *(v for v, _, _ in anchors)) * 1.06
        bins = np.linspace(0, hi, 46)

        a.axvspan(0, today, color="0.55", alpha=0.15, lw=0, zorder=0)
        a.hist(latent, bins=bins, color=LIGHT_GRAY, alpha=0.55, edgecolor="none")
        a.hist(confirmed, bins=bins, color=BLUE, alpha=0.85, edgecolor="white", lw=0.3)
        a.axvline(today, color=RED, lw=1.9)

        ymax = a.get_ylim()[1] * 1.24
        a.set_ylim(0, ymax)
        a.annotate(
            "2 Aug 2026",
            (today, ymax * 0.95),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.5,
            color=RED,
            fontweight="bold",
        )

        for j, (x, label, ls) in enumerate(anchors):
            a.axvline(x, color="0.25", ls=ls, lw=1.1)
            side = -1 if x > hi * 0.72 else 1
            a.annotate(
                label,
                (x, ymax * (0.88 - 0.16 * j)),
                xytext=(4.5 * side, 0),
                textcoords="offset points",
                ha="left" if side > 0 else "right",
                va="top",
                fontsize=7,
                color="0.2",
                bbox={"fc": "white", "ec": "none", "alpha": 0.85, "pad": 1},
            )

        a.set_xlim(0, hi)
        a.set_title(title)
        a.set_xlabel(xlabel)
        a.set_ylabel("Posterior draws")
        a.xaxis.set_major_formatter(FuncFormatter(_kfmt))
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(alpha=0.25)

    ax[1].legend(
        handles=[
            Patch(facecolor=BLUE, alpha=0.85, edgecolor="white", label="Confirmed"),
            Patch(facecolor=LIGHT_GRAY, alpha=0.55, edgecolor="none", label="Latent"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=7.5,
        handlelength=1.3,
    )

    fig.tight_layout()
    _save(fig, "fig03_status_quo.png")


def fig_speed() -> None:
    src = json.loads((DATA / "time_avertable.json").read_text())
    den = json.loads((DATA / "avertable.json").read_text())["denominator"][
        "deaths_still_avertable"
    ]["p50"]
    budgets = [5, 10, 20, 35, 50, 100]
    colors = plt.get_cmap("Reds")(np.linspace(0.32, 0.94, len(budgets)))

    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    for b, c in zip(budgets, colors):
        key = f"local_b{b}"
        days = np.array(sorted(int(d) for d in src[key] if int(d) <= 90))
        vals = [src[key][str(d)] for d in days]
        med = np.array([v["deaths_p50"] for v in vals])
        lo = np.array([v["deaths_p25"] for v in vals])
        hi = np.array([v["deaths_p75"] for v in vals])
        ax.fill_between(days, lo, hi, color=c, alpha=0.14, lw=0)
        ax.plot(days, med, "o-", color=c, lw=1.8, ms=4.2, label=f"{b}")

    ax.set_xlabel("Deployment delay after 2 Aug 2026 (days)")
    ax.set_ylabel("Deaths averted")
    ax.set_xlim(-3, 93)
    ax.set_ylim(bottom=-80)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")

    ax2 = ax.secondary_yaxis(
        "right",
        functions=(lambda y: 100 * y / den, lambda y: y * den / 100),
    )
    ax2.set_ylabel("Future deaths averted (%)")
    ax2.spines["top"].set_visible(False)
    ax.legend(
        title="Labs",
        ncol=2,
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        frameon=False,
        columnspacing=0.9,
        handlelength=1.4,
    )
    fig.tight_layout()
    _save(fig, "fig04_speed.png")


def fig_placement_zones() -> None:
    place = json.loads((DATA / "place_avertable_d30_pct_deaths.json").read_text())
    budgets = [5, 10, 20, 35, 50, 75, 100]
    replay = np.genfromtxt(DATA / "new_zone_replay_d30_local.csv", delimiter=",", names=True)
    zstats = {}
    for b in [0] + budgets:
        sel = replay[replay["budget"] == b]
        zstats[b] = _q(sel["new_resid"])

    fig, ax = plt.subplots(1, 2, figsize=(7.1, 3.15))
    g = [place[f"local_b{b}"]["greedy"] for b in budgets]
    r = [place[f"local_b{b}"]["random"] for b in budgets]
    ax[0].fill_between(
        budgets,
        [x["p25"] for x in g],
        [x["p75"] for x in g],
        color=LIGHT_RED,
        alpha=0.35,
        lw=0,
    )
    ax[0].plot(budgets, [x["p50"] for x in g], "o-", color=RED, lw=1.8, ms=4.2)
    ax[0].fill_between(
        budgets,
        [x["p25"] for x in r],
        [x["p75"] for x in r],
        color=LIGHT_GRAY,
        alpha=0.55,
        lw=0,
    )
    ax[0].plot(budgets, [x["p50"] for x in r], "o-", color=GRAY, lw=1.8, ms=4.2)
    ax[0].text(62, 28.5, "model-placed", color=RED, fontsize=8)
    ax[0].text(67, 7.2, "random", color=GRAY, fontsize=8)
    ax[0].set_ylabel("Future deaths averted (%)")
    ax[0].set_title("A. Deaths averted")

    q = [zstats[b] for b in budgets]
    ax[1].axhline(zstats[0]["p50"], color="0.25", lw=1.2, ls=":", label="status quo median")
    ax[1].fill_between(
        budgets,
        [x["p5"] for x in q],
        [x["p95"] for x in q],
        color="#b7d7b0",
        alpha=0.30,
        lw=0,
        label="90% interval",
    )
    ax[1].fill_between(
        budgets,
        [x["p25"] for x in q],
        [x["p75"] for x in q],
        color="#b7d7b0",
        alpha=0.65,
        lw=0,
        label="50% interval",
    )
    ax[1].plot(
        budgets,
        [x["p50"] for x in q],
        "o-",
        color=GREEN,
        lw=1.8,
        ms=4.2,
        label="model-placed median",
    )
    ax[1].set_ylabel("New health zones affected")
    ax[1].set_title("B. Spatial spread")
    ax[1].legend(loc="upper right", frameon=False, fontsize=8)

    for a in ax:
        a.set_xlabel("Lab budget, deployed after 30 days")
        a.set_xlim(0, 105)
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
        a.grid(alpha=0.25)

    fig.tight_layout()
    _save(fig, "fig05_placement_zones.png")


def fig_placement_map() -> None:
    gdf = gpd.read_file(DATA / "placement_map.geojson")
    meta = json.loads((DATA / "placement_map_metadata.json").read_text())
    mapped = gdf[gdf["in_model"]]
    bounds = mapped.total_bounds if len(mapped) else gdf.total_bounds
    pad = 0.035 * max(bounds[2] - bounds[0], bounds[3] - bounds[1])
    bounds = (bounds[0] - pad, bounds[1] - pad, bounds[2] + pad, bounds[3] + pad)

    positive = gdf["sq"].to_numpy()
    positive = positive[positive > 0]
    norm = mcolors.LogNorm(vmin=1, vmax=max(float(np.percentile(positive, 99)), 2.0))
    panels = [
        ("sq", "existing_lab", f"A. Status quo\n{meta['status_quo_zones']} future zones"),
        (
            "random",
            "random_lab",
            f"B. Random {meta['budget']} labs\n{meta['random_zones']} future zones",
        ),
        (
            "celf",
            "celf_lab",
            f"C. CELF {meta['budget']} labs\n{meta['celf_zones']} future zones",
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.75))
    for ax, (val_col, lab_col, title) in zip(axes, panels):
        gdf.plot(ax=ax, facecolor=MAP_EMPTY, edgecolor=MAP_EMPTY_EDGE, lw=0.10)
        sub = gdf[gdf[val_col] > 0]
        if len(sub):
            sub.plot(
                ax=ax,
                column=val_col,
                cmap=MAP_CMAP,
                norm=norm,
                edgecolor="white",
                lw=0.03,
            )
        existing_labs = gdf[gdf["existing_lab"]]
        if len(existing_labs):
            existing_labs.plot(
                ax=ax,
                facecolor="none",
                edgecolor=MAP_EXISTING_LAB,
                lw=0.55,
            )
        new_labs = gdf[gdf[lab_col]]
        if len(new_labs):
            new_labs.plot(ax=ax, facecolor="none", edgecolor=MAP_LAB, lw=0.65)
        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_title(title, fontsize=8.5, fontweight="bold")
        ax.axis("off")

    axes[0].legend(
        handles=[
            Patch(facecolor=MAP_EMPTY, edgecolor=MAP_EMPTY_EDGE, label="0 future infections"),
            Line2D([0], [0], color=MAP_EXISTING_LAB, lw=1.2, label="Existing PCR lab"),
            Line2D([0], [0], color=MAP_LAB, lw=1.2, label="New PCR lab"),
        ],
        fontsize=6.5,
        loc="lower left",
        frameon=True,
        framealpha=0.9,
        borderpad=0.4,
        handlelength=1.3,
    )

    sm = plt.cm.ScalarMappable(cmap=MAP_CMAP, norm=norm)
    sm.set_array([])
    cax = fig.add_axes([0.925, 0.17, 0.018, 0.62])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Future infections per health zone", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    fig.subplots_adjust(left=0.01, right=0.89, top=0.82, bottom=0.02, wspace=0.03)
    fig.text(
        0.01,
        0.95,
        f"Illustrative posterior draw {meta['draw']}; 30-day local-PCR deployment",
        fontsize=7.5,
        color="0.25",
    )
    _save(fig, "fig06_placement_map.png")


def fig_rdt() -> None:
    src = json.loads((DATA / "timerdt_avertable.json").read_text())
    den = json.loads((DATA / "avertable.json").read_text())["denominator"][
        "deaths_still_avertable"
    ]["p50"]
    arms = [
        ("_none", "PCR labs only", GRAY),
        ("_mach75", "+ RDT at lab zones (75%)", "#f26d4b"),
        ("_all50", "+ RDT everywhere (50%)", "#6baed6"),
        ("_all75", "+ RDT everywhere (75%)", "#2171b5"),
    ]

    fig, ax = plt.subplots(figsize=(6.9, 3.55))
    for arm, label, color in arms:
        key = f"local_b20{arm}"
        days = np.array(sorted(int(d) for d in src[key] if int(d) <= 90))
        vals = [src[key][str(d)] for d in days]
        med = np.array([v["deaths_p50"] for v in vals])
        lo = np.array([v["deaths_p25"] for v in vals])
        hi = np.array([v["deaths_p75"] for v in vals])
        ax.fill_between(days, lo, hi, color=color, alpha=0.15, lw=0)
        ax.plot(days, med, "o-", color=color, lw=1.8, ms=4.2, label=label)

    ax.set_xlabel("Deployment delay after 2 Aug 2026 (days)")
    ax.set_ylabel("Deaths averted")
    ax.set_xlim(-3, 93)
    ax.set_ylim(bottom=-80)
    ax.spines["top"].set_visible(False)
    ax.grid(alpha=0.25)
    ax.yaxis.set_major_formatter(lambda v, _pos: f"{v:,.0f}")

    ax2 = ax.secondary_yaxis(
        "right",
        functions=(lambda y: 100 * y / den, lambda y: y * den / 100),
    )
    ax2.set_ylabel("Future deaths averted (%)")
    ax2.spines["top"].set_visible(False)
    ax.legend(
        title="20 labs plus RDT",
        fontsize=8,
        title_fontsize=8,
        loc="upper right",
        frameon=False,
        handlelength=1.6,
    )
    fig.tight_layout()
    _save(fig, "fig07_rdt.png")


def fig_appendix_pairplot() -> None:
    post = np.load(DATA / "sq_draws.npz")
    keys = [
        ("R0", r"$R_0$"),
        ("R_floor", r"$R_{\mathrm{floor}}$"),
        ("cum_half", r"$D_{1/2,z}$"),
        ("jump_prob", r"$m$"),
        ("ascertainment", r"$p_{\mathrm{PCR,lab}}$"),
        ("ifr", "IFR"),
    ]
    x = np.column_stack([np.asarray(post[k], float) for k, _ in keys])
    n = len(keys)
    rng = np.random.default_rng(20260802)
    pick = rng.choice(x.shape[0], size=min(900, x.shape[0]), replace=False)
    xs = x[pick]

    fig, axes = plt.subplots(n, n, figsize=(7.1, 7.1))
    for r in range(n):
        for c in range(n):
            ax = axes[r, c]
            if r == c:
                ax.hist(x[:, c], bins=28, color=BLUE, alpha=0.82, edgecolor="none")
                q5, q50, q95 = np.percentile(x[:, c], [5, 50, 95])
                ax.axvline(q50, color="white", lw=1.2)
                ax.set_title(f"{keys[c][1]}\n{q50:.2g} [{q5:.2g}, {q95:.2g}]", fontsize=7.2)
            elif r > c:
                ax.scatter(xs[:, c], xs[:, r], s=3, color=BLUE, alpha=0.20, lw=0)
            else:
                corr = float(np.corrcoef(x[:, c], x[:, r])[0, 1])
                ax.text(
                    0.5,
                    0.5,
                    f"$r={corr:.2f}$",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color="0.25",
                    fontsize=8,
                )
            if r < n - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(keys[c][1], fontsize=8)
            if c > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(keys[r][1], fontsize=8)
            ax.tick_params(labelsize=6, length=2, pad=1)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout(h_pad=0.25, w_pad=0.25)
    _save(fig, "figA01_posterior_pairplot.png")


def _band(ax, x, arr, color, label):
    q5, q50, q95 = np.percentile(arr, [5, 50, 95], axis=0)
    ax.fill_between(x, q5, q95, color=color, alpha=0.22, lw=0, label=f"{label} 5-95%")
    ax.plot(x, q50, color=color, lw=1.8, label=f"{label} median")


def _style_time_axis(ax):
    loc = mdates.AutoDateLocator(minticks=3, maxticks=5)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.tick_params(axis="x", rotation=35)


def fig_appendix_ppc() -> None:
    ppc = np.load(DATA / "posterior_predictive.npz")

    dates = ppc["dates"].astype("datetime64[D]")
    conf_mid = ppc["conf_mid"].astype("datetime64[D]")
    death_mid = ppc["death_mid"].astype("datetime64[D]")
    iso_dates = ppc["iso_dates"].astype("datetime64[D]")
    zones = ppc["zones"]

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
        }
    )
    fig, axes = plt.subplots(3, 2, figsize=(7.1, 6.9))

    ax = axes[0, 0]
    _band(ax, dates, ppc["nat_pp"], "#2b6cb0", "Model")
    ax.plot(dates, ppc["obs_nat_cum"], "k.-", ms=2.2, lw=1.0, label="Observed")
    ax.set_title("Cumulative confirmed")
    ax.set_ylabel("Cases")
    ax.legend(loc="upper left", frameon=False)
    _style_time_axis(ax)

    ax = axes[0, 1]
    _band(ax, conf_mid, ppc["conf_new_pp"], "#dd6b20", "Model")
    ax.plot(conf_mid, ppc["obs_conf_new"], "k.-", ms=2.2, lw=1.0, label="Observed")
    ax.set_title("3-day confirmed incidence")
    ax.set_ylabel("Cases / 3 d")
    ax.legend(loc="upper left", frameon=False)
    _style_time_axis(ax)

    ax = axes[1, 0]
    _band(ax, death_mid, ppc["deaths_pp"], "#805ad5", "Model")
    ax.plot(death_mid, ppc["obs_deaths_binned"], "k.-", ms=2.2, lw=1.0, label="Observed")
    ax.plot(death_mid[-1], ppc["obs_deaths_binned"][-1], "v", color="red", ms=5)
    ax.set_title("Confirmed deaths")
    ax.set_ylabel("Deaths / cell")
    ax.legend(loc="upper left", frameon=False)
    _style_time_axis(ax)

    ax = axes[1, 1]
    _band(ax, iso_dates, ppc["conf_iso_pp"], "#2b6cb0", "Model")
    ax.plot(
        iso_dates,
        ppc["obs_confirmed_iso"],
        "k.-",
        ms=2.2,
        lw=1.0,
        label="Observed",
    )
    ax.set_title("Confirmed-case isolation census")
    ax.set_ylabel("People")
    ax.legend(loc="upper left", frameon=False)
    _style_time_axis(ax)

    ax = axes[2, 0]
    obs_z = ppc["obs_zone_final"]
    sel = np.argsort(-obs_z)[:15]
    q5, q50, q95 = np.percentile(ppc["zone_pp"][:, sel], [5, 50, 95], axis=0)
    x = np.arange(sel.size)
    ax.bar(x - 0.2, obs_z[sel], width=0.38, color="0.25", label="Observed")
    ax.bar(
        x + 0.2,
        q50,
        width=0.38,
        yerr=[q50 - q5, q95 - q50],
        color="#3182bd",
        ecolor="#08519c",
        capsize=2,
        alpha=0.9,
        label="Model median",
    )
    ax.set_title("Cumulative confirmed by health zone")
    ax.set_ylabel("Cases")
    ax.set_xticks(x)
    ax.set_xticklabels([zones[i] for i in sel], rotation=65, ha="right")
    ax.legend(loc="upper right", frameon=False)

    ax = axes[2, 1]
    _band(ax, dates, ppc["lat_pp"], "#2f9e44", "Model latent")
    ax.errorbar(
        [np.datetime64(str(ppc["epi_date"]))],
        [float(ppc["epi_mean"])],
        yerr=[
            [float(ppc["epi_mean"]) - float(ppc["epi_lo"])],
            [float(ppc["epi_hi"]) - float(ppc["epi_mean"])],
        ],
        fmt="o",
        color="red",
        capsize=3,
        ms=4,
        label="Latent anchor",
    )
    ax.set_title("Latent burden")
    ax.set_ylabel("Infections")
    ax.legend(loc="upper left", frameon=False)
    _style_time_axis(ax)

    for ax in axes.ravel():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.22, lw=0.5)

    fig.tight_layout()
    _save(fig, "figA02_posterior_predictive.png")


def main() -> None:
    fig_response_functions()
    fig_status_quo()
    fig_speed()
    fig_placement_zones()
    fig_placement_map()
    fig_rdt()
    fig_appendix_pairplot()
    fig_appendix_ppc()


if __name__ == "__main__":
    main()
