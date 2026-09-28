"""Stochastic spatial branching-process model of BDBV spread with a
capacity-driven detection/confirmation layer.

Only calibration-path code lives here. The intervention tree/prune
machinery used by the forward scenarios lives in `tree.py`.

Structural assumptions baked in (previously behind toggles):
- brake_form = "sigmoid" (Hill brake in death-proxy units, lagged)
- ramp_kernel on (death-ramped infectiousness + burial window)
- nuanced_response on (competing PCR + clinical detection channels)
- collection_local on (PCR is the shared prerequisite for iso + confirm)
- behaviour_recognition_gated on (no brake pre-recognition)
- awareness_enabled on (non-machine zones "awaken" post first local confirm)
- response_start_day always set (calibration always has one)
- p_access = 1.0 (all served-zone cases reach the local lab)

Detection and confirmation detail:
- Time-varying unserved turnaround (Kinshasa-bulk pre-2026-05-30,
  regional-hub 5d post) via `unserved_switch_day` + two turnaround values.
- Cadaver swab channel (PCR-based, safe-burial-team collection):
  `cadaver_swab_*` fields (renamed from `rdt_*`). Decoupled:
  reached bodies get safe burial at t_death (transmission cut);
  positive PCR + delay adds them to the confirmed pool.
- `cadaver_swab_delay_days` — swab-to-confirmation delay (~5d PCR turnaround).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom as _nbinom

REF_DATE = pd.Timestamp("2026-01-01")


def to_day(ts) -> int:
    return int((pd.Timestamp(ts) - REF_DATE).days)


def from_day(d: int) -> pd.Timestamp:
    return REF_DATE + pd.Timedelta(days=int(d))


# --------------------------------------------------------------------------
# Params
# --------------------------------------------------------------------------


@dataclass
class Params:
    """One prior draw. Times are days from REF_DATE (calendar) or since
    infection (case-time), as noted per field. Fields grouped for readability
    but ordered non-default-first per dataclass requirement."""

    # ---- Non-default (required) fields ----
    R0: float  # intrinsic reproduction number (pre-brake)
    k: float  # NB offspring dispersion
    gi_shape: float  # generation interval Gamma(shape, scale), mean ~15.3d
    gi_scale: float  # (kept for older call sites — the ramp kernel is always used)
    incub_meanlog: float  # LogNormal incubation, mean ~6.3d
    incub_sdlog: float
    jump_prob: float  # m = P(offspring leaves parent zone)
    ifr: float  # infection fatality ratio
    report_delay_mean: float  # legacy RNG-stream draw; fixed turnarounds are used
    response_start_day: int  # recognition day; isolation begins here
    seed_n: int  # initial infections
    seed_day: int  # seed day (from REF_DATE)
    seed_zone: str  # seed zone name

    # ---- Defaulted fields ----
    # death-ramped kernel + burial
    ramp_a: float = 1.0
    w_burial: float = 0.15
    burial_width: float = 0.8
    otd_shape: float = 2.25  # onset->death Gamma(shape, scale), mean ~12d
    otd_scale: float = 5.33
    recovery_mean: float = 16.0
    recovery_sd: float = 6.0
    # Hill (sigmoid) brake — LOCAL (per-zone) driver
    R_floor: float = 0.0
    cum_half: float = 0.0
    hill_n: float = 1.0
    brake_lag_days: float = 0.0
    # NATIONAL awareness brake. A second Hill factor on the same visible-death units,
    # driven by the NATIONAL post-recognition toll rather than the parent's own zone.
    # Multiplies the local brake, so a newly invaded zone inherits the country's awareness
    # state instead of restarting naive — without this, forward runs let each newly
    # invaded zone rediscover the epidemic from scratch and burden runs away.
    # nat_half = 0.0 leaves the factor at 1.0, i.e. the brake is off.
    nat_half: float = 0.0  # national visible deaths (ifr·cum) at which the brake halves
    nat_hill_n: float = 2.0  # steepness; gentler than the local hill_n by default
    nat_brake_driver: str = "deaths"  # "deaths" (endogenous) | "calendar" (exogenous)
    nat_half_days: float = 0.0  # "calendar" driver: days past response_start at half-brake
    # FLOOR on the national factor, B_nat' = nat_floor + (1-nat_floor)*B_nat. This is the
    # share of frontier transmission national awareness cannot reach — chains driven by
    # people who have not encountered the response (remote zones, cross-border and mining
    # movement, unrecognised chains ahead of surveillance). Without a floor, and because the
    # two factors MULTIPLY, a national factor of 0.43 at the horizon takes a 40% haircut off
    # a zone that has seen nothing, which drives realized Rt well below the observed band.
    # Local experience still brakes fully once a zone is invaded — B_local is untouched.
    nat_floor: float = 0.0  # in [0, 1); 0 = no floor
    # ---- WHERE the national factor multiplies ----------------------------------------------
    #
    #   "gap"    (the older form)
    #       r_eff = m_z * [ R_floor + (R0 - R_floor) * B_loc * B_nat ]
    #   "global" (used here)
    #       r_eff = m_z * S_nat(t) * [ R_floor + (R0 - R_floor) * B_loc ]
    #
    # Under "gap" the national factor multiplies the GAP, so it can only move the naive
    # (unbraked) endpoint and vanishes on a fully-braked zone; R_floor is then the sole
    # late-time asymptote, which welds in-window fit and forward safety to one parameter.
    # Under "global" it multiplies the WHOLE r_eff, so the asymptote becomes S_min*R_floor
    # and a supercritical LOCAL floor no longer implies a permanently supercritical zone.
    #
    # S_nat is a CLOCK, not a feedback loop — it stands for staff, therapeutics, vaccine and
    # distributed diagnostics arriving on a calendar rather than in proportion to deaths:
    #
    #   el      = max(t - brake_lag_days - response_start_day - nat_scale_t0, 0)   CLAMPED at 0
    #   S_raw   = 1 / (1 + (el / nat_scale_T) ** nat_scale_n)
    #   S_nat   = nat_scale_min + (1 - nat_scale_min) * S_raw
    #
    # The clamp is what makes S_nat == 1.0 EXACTLY at and before ONSET; without it an even
    # exponent would give S < 1 on both sides of onset (a national response acting before it
    # starts). This is the same Hill the "calendar" gap-driver uses, and it is a sigmoid with
    # precisely the four properties we want (writing t_on = t_resp + nat_scale_t0):
    #
    #   ceiling  S_nat(t <= t_on)          = 1                        exactly, by the clamp
    #   centre   S_nat(t_on + nat_scale_T) = (1 + S_min)/2            exactly, 50% point
    #   floor    S_nat(t -> inf)           = nat_scale_min
    #   slope at centre  |dS/dt|           = nat_scale_n * (1 - S_min) / (4 * nat_scale_T)
    #
    # (A logistic cannot have all four: forcing the 50% point at T leaves S(t_resp) =
    # 1 - (1-S_min)/(1+e^{nT/... }) < 1, a ~12% downward jump at t_resp for the n=2-equivalent
    # slope. The Hill's inflection sits at 0.577*T rather than at T, which is the only price.)
    #
    # nat_brake_placement = "gap" leaves every line below untouched.
    nat_brake_placement: str = "gap"  # "gap" | "global"
    nat_scale_T: float = 0.0  # days past ONSET at the 50% point; 0 = off
    nat_scale_min: float = 1.0  # S_min, the late-time national multiplier; 1 = off
    nat_scale_n: float = 2.0  # Hill exponent; sets the slope at the centre
    # ONSET OFFSET. The clock starts at `response_start_day + nat_scale_t0` rather than at
    # response start; t0 = 0 starts it at response start.
    #
    # The shipped config sets t0 = 88 = the horizon (2026-08-02), which makes the clock
    # EXACTLY invisible to the fit rather than merely near-invisible. The bound: `nat_scale` is read at
    # thr = t_infect - brake_lag_days (16 d), and the fit only ever evaluates parents with
    # t_infect <= 88, so any t0 >= 88 - 16 = 72 gives S_nat == 1 at every in-window r_eff
    # evaluation. t0 = 88 takes the full 16 d of margin and puts onset on a DATE rather than
    # on an offset chosen to just clear the bound.
    #
    # Forward runs, with their longer horizon, engage the clock normally past day
    # t_resp + t0 + brake_lag. That asymmetry is the design.
    nat_scale_t0: float = 0.0  # days past response_start at which the clock STARTS
    # WANING MEMORY. Replaces the cumulative brake driver with an
    # exponentially-weighted one, shared by the local and national factors:
    #
    #   D_w(t) = ifr * sum_j exp(-(t - lag - t_j) / tau_w)    over post-recognition j, t_j <= t-lag
    #
    # tau_w = inf recovers the plain cumulative driver D_cum EXACTLY. Finite tau_w drops both
    # irreversibility and timing-blindness together: risk salience decays with half-life
    # ln2 * tau_w, so the brake tightens while deaths are current and RELAXES when they stop.
    # That is what lets an established zone come off the floor without deleting the national
    # awareness mechanism that keeps forward burden finite.
    #
    # NOTE ON THE DECAY ORIGIN. The natural way to write the weight is as exp(-(t - t_j)/tau_w) with t_j an INFECTION time; the code decays from
    # t_j + lag, the day that infection becomes VISIBLE as a death. The two differ by the constant
    # exp(-lag/tau_w), which is tau_w-dependent and would therefore confound the tau_w sweep with
    # a pure rescaling of cum_half. Decaying from visibility is also the mechanism as stated
    # ("salience of a death decays from when it is seen"), and it keeps the freshest visible death
    # at weight 1 so D_w and D_cum share units at short times. Both forms give D_cum as tau_w->inf.
    #
    # Units warning (README's "implementation trap"): at steady infection rate lambda, D_w ->
    # ifr*lambda*tau_w, a rate x time, NOT a count. cum_half / nat_half priors are re-derived on
    # measured units before any fit.
    tau_w: float = float("inf")  # days; inf = the plain cumulative driver
    # lab turnaround — three tiers via time-varying unserved switch
    local_turnaround: float = 1.0  # served zones (within 3-hr catchment of online lab)
    unserved_turnaround_early: float = 8.0  # pre-switch: samples ship to Kinshasa
    unserved_turnaround_late: float = 5.0  # post-switch: samples route to regional hub
    unserved_switch_day: int = 10_000_000  # calendar day of routing switch (default +inf = never)
    # collection timing
    collect_delay_machine: float = 2.0
    collect_delay_nomachine: float = 4.0
    # tri-state PCR ladder
    p_pcr_high: float = 1.0  # machine HZ
    p_pcr_mid: float = 1.0  # aware non-machine
    p_pcr_low: float = 1.0  # naive non-machine
    aware_threshold: int = 1  # # local confirms needed to "awaken"
    # clinical detection
    p_clin: float = 0.25
    clin_beta_a: float = 2.0
    clin_beta_b: float = 1.0
    # isolation (nested two-tier)
    p_iso_presump: float = 0.25
    p_iso_confirmed: float = 0.85
    p_iso_clinical: float | None = None
    # cadaver swab channel (status quo = post-mortem PCR, ~0.90 sens, delayed)
    cadaver_swab_coverage: float = 0.0
    cadaver_swab_sens: float = 0.90
    cadaver_swab_delay_days: float = 5.0
    cadaver_swab_scope_all: bool = True
    # SQ contact tracing
    sq_trace_eff: float = 0.0
    sq_trace_lag: float = 2.0
    # contact-LISTING throughput cap. The response lists a roughly
    # constant number of contacts per day (435 +/- 186, uncorrelated with same-day cases,
    # r = 0.13), so contacts listed per case collapses as incidence rises (13.3 on
    # <=40-case days vs 8.6 on >40). Effective interception is therefore
    #     sq_trace_eff * min(1, trace_cap / L(t)),   L(t) = confirmations in the last
    #                                                       trace_window days (national)
    # evaluated at the PARENT's confirmation day, because that is when a case's contacts
    # get listed. trace_cap = inf is a no-op (the isfinite guard also leaves the RNG stream
    # untouched). NOT calibrated: measured at 435/13.3 ~= 33 cases/day ~= 460 per 14 d.
    trace_cap: float = float("inf")  # confirmations per trace_window that can be fully listed
    trace_window: float = 14.0  # days; FIXED (a window length is not identifiable today)
    # Bed-stay caps
    bed_stay_suspect_days: float = float("inf")
    bed_stay_post_confirm_days: float = float("inf")
    # care-seeking referral to CTE at care-seeking time
    f_refer: float = 0.0  # P(referring-zone case relocates to CTE at t_collect)
    noso_boost: float = 0.0  # brake-independent R excess for referred cases at CTE (× R0)
    # Seed conditioning: condition the index case's offspring count.
    # None  -> unconditioned draw.
    # "nonzero" -> exact zero-truncation: u ~ U(F(0), 1) with F(0) = (k/(k+R0))^k, so the seed
    #              is conditioned on infecting >= 1 person. Parameter-free and data-independent.
    # float -> a hand-picked uniform floor, for sensitivity probes only, NOT for fitting.
    # A Params FIELD, not a module global: a global does not survive a spawn-started worker
    # pool, which is how every fit here runs.
    seed_u_min: str | float | None = None

    def __post_init__(self) -> None:
        # The waning driver's prefix sum stores exp((t - t0)/tau_w), which
        # overflows float64 once (t - t0)/tau_w > 709. The longest span this model is asked for is
        # a +540 d projection past a ~215 d horizon, ~700 d from response start, so the bound bites
        # only below tau_w ~= 1 d. Refuse there rather than silently returning inf/nan drivers —
        # a sub-day behaviour memory is not a value any prior should be proposing anyway.
        if np.isfinite(self.tau_w) and self.tau_w < 1.0:
            raise ValueError(
                f"tau_w must be >= 1 day (or inf for the cumulative driver); got {self.tau_w}"
            )
        # The two national mechanisms are alternatives, not layers. Under "global" the
        # deaths-driven gap factor is REPLACED, so a config that still carries nat_half/nat_floor
        # is a config whose author has not decided which one is live — refuse it rather than
        # silently running both and reporting a floor whose meaning changed.
        if self.nat_brake_placement not in ("gap", "global"):
            raise ValueError(
                f"nat_brake_placement must be 'gap' or 'global'; got {self.nat_brake_placement!r}"
            )
        if self.nat_brake_placement == "global":
            if self.nat_half > 0 or self.nat_floor > 0 or self.nat_half_days > 0:
                raise ValueError(
                    "nat_brake_placement='global' replaces the gap-brake; set "
                    "nat_half / nat_floor / nat_half_days to 0"
                )
            if not (0.0 <= self.nat_scale_min <= 1.0):
                raise ValueError(f"nat_scale_min must be in [0, 1]; got {self.nat_scale_min}")
        # A negative onset would start the clock BEFORE the response, which is the one
        # thing the clamp exists to forbid.
        if self.nat_scale_t0 < 0.0:
            raise ValueError(f"nat_scale_t0 must be >= 0; got {self.nat_scale_t0}")


def _decayed_driver(ts: np.ndarray, thr: np.ndarray, tau_w: float, t0: float) -> np.ndarray:
    """The brake driver, in either functional form.

    `ts` is a SORTED array of post-recognition infection times; `thr` the query times
    (parent infection time minus `brake_lag_days`). Returns, per query,

        tau_w = inf     count of { t_j <= thr }                        <- plain cumulative
        tau_w finite    sum_{t_j <= thr} exp(-(thr - t_j) / tau_w)     <- waning memory

    Decay runs from `t_j` — the infection time — but the sum is restricted to `t_j <= thr`, so
    the freshest contributing entry carries weight ~exp(-0) = 1 and the two forms share units at
    short times. See the `Params.tau_w` note for why the origin is visibility rather than the
    README's literal `t`.

    Call sites pass sorted per-zone or national infection-time arrays and pay
    the direct search cost when they read the driver.
    """
    if not np.isfinite(tau_w):
        return np.searchsorted(ts, thr, side="right").astype(float)
    # Empty registry -> zero driver. `np.where` evaluates BOTH branches, so the clamped index
    # below is still taken even where the count is 0, and on a size-0 `wcum` that raises. The
    # infinite branch above never hits this (searchsorted is total), which is why every experiment
    # in the cumulative-driver limit. The per-zone call site guards `ts.size > 0`;
    # the NATIONAL one cannot, because it must return a value for every parent in the generation,
    # and `tinf_post_nat` is empty for every generation before the first post-recognition
    # infection. Guarding here rather than at the call site keeps the function total.
    if ts.size == 0:
        return np.zeros(np.shape(thr), dtype=float)
    idx = np.searchsorted(ts, thr, side="right")
    wcum = np.cumsum(np.exp((ts - t0) / tau_w))
    return np.where(idx > 0, np.exp(-(thr - t0) / tau_w) * wcum[np.maximum(idx - 1, 0)], 0.0)


def nat_scale(thr: np.ndarray, params: Params) -> np.ndarray:
    """`S_nat`, the SLOW national multiplier on the whole of `r_eff`.

    `thr` is the lagged query time (`t_infect - brake_lag_days`), the same clock the local brake
    reads, so `S_nat` and `B_loc` are lagged identically. See `Params.nat_brake_placement` for the
    form and for why the elapsed time is clamped at zero.

    The onset offset `nat_scale_t0` returns a HARD 1.0 wherever the clock has not yet engaged.
    `S_min + (1 - S_min) * 1.0` is exactly 1.0 in IEEE754 for every S_min tested, but making
    the clock invisible in-window rests on the pre-onset no-op being exact, so it is asserted
    structurally rather than inherited from a floating-point identity.

    This helper is shared by the slow national response and local brake clocks.
    """
    el = np.maximum(thr - params.response_start_day - params.nat_scale_t0, 0.0)
    if params.nat_scale_T > 0:
        s = 1.0 / (1.0 + np.power(el / params.nat_scale_T, params.nat_scale_n))
    else:
        s = np.ones(np.shape(thr), dtype=float)  # T = 0 -> clock never engages
    return np.where(el > 0.0, params.nat_scale_min + (1.0 - params.nat_scale_min) * s, 1.0)


# --------------------------------------------------------------------------
# Lab catchment (unchanged from central model.py)
# --------------------------------------------------------------------------


def build_served_from_day(
    zones: list[str],
    travel_time: pd.DataFrame,
    lab_schedule: list[dict],
    catchment_minutes: float,
) -> np.ndarray:
    """Earliest day each zone is served by an online lab within `catchment_minutes`
    road time. Zones never served get +inf. `lab_schedule` = [{zone_nom, online_day}, ...]
    excluding Kinshasa (which is the baseline)."""
    served = np.full(len(zones), np.inf)
    for lab in lab_schedule:
        lz = lab["zone_nom"]
        online = lab["online_day"]
        if lz not in travel_time.index:
            continue
        for zi, z in enumerate(zones):
            if z not in travel_time.columns:
                continue
            tt = travel_time.at[lz, z]
            if pd.notna(tt) and tt <= catchment_minutes:
                served[zi] = min(served[zi], online)
    return served


# --------------------------------------------------------------------------
# Confirmation-capacity FIFO layer (unchanged)
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# Tracing diagnostics. OFF by default, and with RECORD_TRACE_DIAG False the calibration path
# is byte-identical to having no diagnostics at all. Turned on by analysis scripts, which
# need three things the returned DataFrame cannot express:
#   * the LOAD the model actually saw (causal, generation-by-generation) vs the completed
#     simulation's load  -> A2 load fidelity
#   * the realized coverage PATH, since a time-average hides the mechanism entirely
#   * interception attributed to the trace channel, the quantity with an empirical ceiling
#     (12-16 % from the SitReps)
# --------------------------------------------------------------------------
RECORD_TRACE_DIAG = False
# Brake-driver completeness probe. Records, per case, the cumulative post-recognition
# in-zone infection count the Hill brake ACTUALLY used, so it can be compared against the count
# recomputed from the finished run. Off by default; costs nothing when off.
RECORD_ISO_DIAG = False
ISO_DIAG: dict = {"case": [], "off": []}
RECORD_BRAKE_DIAG = False
BRAKE_DIAG: dict = {"rows": []}
TRACE_DIAG: dict = {}


def trace_diag_reset():
    TRACE_DIAG.clear()
    TRACE_DIAG.update(
        {
            "load": [],
            "conf_days": [],
            "n_listed": 0,
            "n_inf": 0,
            "n_conf": 0,
            "n_conf_listed": 0,
            "would_be": 0,
            "averted_trace": 0,
            "by_week": {},
        }
    )


def _td_week(day_arr, mask, col):
    """Accumulate `mask` counts into per-ISO-week slot `col` of TRACE_DIAG["by_week"].
    Slots: 0 n_inf, 1 listed, 2 would_be, 3 averted_by_trace, 4 n_conf, 5 conf_listed."""
    wk = (np.asarray(day_arr) // 7).astype(int)
    for w in np.unique(wk):
        m = (wk == w) & mask
        if m.any():
            r = TRACE_DIAG["by_week"].setdefault(int(w), [0, 0, 0, 0, 0, 0])
            r[col] += int(m.sum())


def _confirm_with_capacity(t_arrive, cap_days, cap_vals, queue_query_days=None):
    """Capacity-limited FIFO confirmation. Samples arrive at t_arrive; lab confirms
    up to cap(day) per day; backlog builds and clears in steps. `cap_days` ascending
    ints, `cap_vals` piecewise constant. Optional `queue_query_days` returns
    end-of-day queue size for scoring/backlog anchors."""
    t_arrive = np.asarray(t_arrive, float)
    n = t_arrive.size
    out = np.full(n, np.inf)
    if n == 0:
        if queue_query_days is None:
            return out
        return out, np.zeros(np.asarray(queue_query_days).size)
    cap_days = np.asarray(cap_days)
    cap_vals = np.asarray(cap_vals, float)
    order = np.argsort(t_arrive, kind="stable")
    arr = t_arrive[order]
    i = 0
    day = int(np.floor(arr[0]))
    guard = day + n + int(cap_days.max() if cap_days.size else 0) + 100000
    q_days = None if queue_query_days is None else np.asarray(queue_query_days, int)
    q_out = None if q_days is None else np.zeros(q_days.size, float)
    q_idx = 0
    while i < n and day <= guard:
        c = (
            cap_vals[max(np.searchsorted(cap_days, day, side="right") - 1, 0)]
            if cap_days.size
            else np.inf
        )
        cnt = 0
        while i < n and arr[i] <= day and cnt < c:
            out[order[i]] = float(day)
            i += 1
            cnt += 1
        if q_days is not None:
            arrived_by_day = int(np.searchsorted(arr, day, side="right"))
            while q_idx < q_days.size and q_days[q_idx] <= day:
                q_out[q_idx] = max(0, arrived_by_day - i)
                q_idx += 1
        day += 1
    if q_days is not None:
        while q_idx < q_days.size:
            q_out[q_idx] = 0.0
            q_idx += 1
        return out, q_out
    return out


# --------------------------------------------------------------------------
# Nuanced iso + burial (with option-(b) decoupled cadaver channel)
# --------------------------------------------------------------------------


def _draw_nuanced_u_tail(rng, n, params):
    """Post-PCR nuanced uniforms (clin, fclin, presump, conf, cadaver, refer). `pcr` is
    drawn upstream. Kept as a helper so the intervention path can replay these
    draws under alternative placement (if they re-introduce tree machinery)."""
    return {
        "clin": rng.random(n),
        "fclin": rng.beta(params.clin_beta_a, params.clin_beta_b, n),
        "presump": rng.random(n),
        "conf": rng.random(n),
        "cadaver": rng.random(n),
        "refer": rng.random(n),  # draw for care-seeking referral
    }


def _nuanced_iso_burial(
    params, n, incub, died_p, otd_p, trec_p, cur_tinf, turnaround, pcr_prob, onsite, local, u
) -> tuple[np.ndarray, ...]:
    """Detection -> isolation -> cadaver-burial pipeline.

    Two living-detection channels: clinical (late, illness-fraction) and PCR
    (result after `turnaround`). Nested two-tier isolation: presumptive at
    p_iso_presump; PCR+ upgrades same latent to p_iso_confirmed TOTAL.

    Cadaver channel (option-(b) decoupled):
    - reached = u["cadaver"] < cadaver_swab_coverage         # safe-burial team gets there
    - positive = u["cadaver"] < cadaver_swab_coverage * cadaver_swab_sens
    - safe burial triggered by REACHED (not by positive test): burial-window
      offspring truncated for reached bodies via `safe_burial` mask.
    - confirmation added for POSITIVE undiagnosed dead cases at
      t_death + cadaver_swab_delay_days.

    Returns iso_cutoff (since infection), safe_burial mask, t_confirm (since
    infection; inf if never confirmed), got_pcr flag, t_presump_eff (for
    isolation census), and end_illness (for iso-end bookkeeping).
    """
    illness = np.where(died_p, otd_p, trec_p)
    end_illness = incub + illness

    # --- living detection ---
    got_pcr = u["pcr"] < pcr_prob
    clin = u["clin"] < params.p_clin
    t_clin = incub + u["fclin"] * illness
    t_collect = incub + np.where(
        onsite, params.collect_delay_machine, params.collect_delay_nomachine
    )

    # No BDBV dx/sampling before recognition; cryptic cases caught AT recognition.
    recog = params.response_start_day - cur_tinf
    t_clin = np.maximum(t_clin, recog)
    t_collect = np.maximum(t_collect, recog)
    clin = clin & (t_clin <= end_illness)
    got_pcr = got_pcr & (t_collect <= end_illness)

    t_result = t_collect + turnaround
    diagnosed_alive = clin | got_pcr

    # --- nested two-tier isolation ---
    iso_cutoff = np.full(n, np.inf)
    if params.p_iso_clinical is None:
        t_presump = np.full(n, np.inf)
        t_presump = np.where(clin, np.minimum(t_presump, t_clin), t_presump)
        t_presump = np.where(got_pcr, np.minimum(t_presump, t_collect), t_presump)
        presump_eff = (clin | got_pcr) & (u["presump"] < params.p_iso_presump)
    else:
        t_presump = np.where(got_pcr, t_collect, np.inf)
        presump_eff = got_pcr & (u["presump"] < params.p_iso_presump)
        clin_eff = clin & (u["clin_iso"] < params.p_iso_clinical)
        iso_cutoff = np.where(clin_eff, np.minimum(iso_cutoff, t_clin), iso_cutoff)

    conf_eff = got_pcr & (u["presump"] < params.p_iso_confirmed)  # NESTED same latent
    iso_cutoff = np.where(presump_eff, np.minimum(iso_cutoff, t_presump), iso_cutoff)
    iso_cutoff = np.where(conf_eff, np.minimum(iso_cutoff, t_result), iso_cutoff)
    # isolation can't bite pre-response
    iso_cutoff = np.where((cur_tinf + iso_cutoff) >= params.response_start_day, iso_cutoff, np.inf)

    # --- cadaver channel (option-(b) decoupled) ---
    cad_here = bool(params.cadaver_swab_scope_all) | local
    # reached = burial team reaches the body (regardless of test result)
    reached = (
        died_p & (~diagnosed_alive) & cad_here & (u["cadaver"] < params.cadaver_swab_coverage)
    )
    # positive = test returns positive (subset of reached)
    cad_positive = (
        died_p
        & (~diagnosed_alive)
        & cad_here
        & (u["cadaver"] < params.cadaver_swab_coverage * params.cadaver_swab_sens)
    )
    # Safe burial happens for BOTH diagnosed-alive AND reached bodies (regardless of test result)
    safe_burial = diagnosed_alive | reached

    # Confirmation: PCR+ for living cases at t_result; positive cadaver swab at
    # t_death + delay. Both since-infection times.
    t_death_case = incub + otd_p  # time of death, since infection (only meaningful if died_p)
    t_confirm_alive = np.where(got_pcr, t_result, np.inf)
    t_confirm_cadaver = np.where(
        cad_positive, t_death_case + params.cadaver_swab_delay_days, np.inf
    )
    t_confirm = np.minimum(t_confirm_alive, t_confirm_cadaver)

    # Isolation census start (presumptive isolation admission), for suspects-in-iso stream
    t_presump_eff = np.where(presump_eff, t_presump, np.inf)
    t_presump_eff = np.where(
        (cur_tinf + t_presump_eff) >= params.response_start_day, t_presump_eff, np.inf
    )

    # return t_collect (since infection) for the referral machinery in simulate()
    return iso_cutoff, safe_burial, t_confirm, got_pcr, t_presump_eff, end_illness, t_collect


# --------------------------------------------------------------------------
# Main simulation (calibration path — no tree recording)
# --------------------------------------------------------------------------


def simulate(
    params: Params,
    zones: list[str],
    mobility_out: np.ndarray,
    served_from_day: np.ndarray,
    horizon_day: int,
    rng: np.random.Generator,
    max_infections: int = 200_000,
    collect_high_from_day: np.ndarray | None = None,
    confirm_capacity: tuple | None = None,
    queue_query_days: np.ndarray | None = None,
    care_destination: np.ndarray | None = None,  # (Z,) int, i -> CTE zone (self if i is CTE)
    zone_r_mult: np.ndarray | None = None,  # (Z,) float multiplier on r_eff per zone
    listing_load: tuple | None = None,  # (days, vals) EXOGENOUS tracing load
) -> pd.DataFrame:
    """One stochastic realisation. Returns per-infection DataFrame with
    zone_idx, t_infect, onset, t_detect, detected, died, t_death, isolated,
    iso_start, iso_end, hit_cap, care_zone.

    Structural assumptions:
    - Sigmoid Hill brake, ramp kernel, nuanced response, collection-local,
      behaviour-recognition-gated, awareness-enabled, response_start_day set.
    - Time-varying unserved turnaround via unserved_switch_day.
    - Care-seeking referral: `f_refer` of cases in a non-CTE zone physically
      relocate to their `care_destination[origin]` at t_collect. Their
      post-collect offspring generated from CTE zone; noso_boost adds
      brake-independent R excess.

    Two further transmission brakes sit on top of that, both no-ops at their
    defaults:
    - NATIONAL awareness (`params.nat_half` > 0): a second Hill factor driven by
      the national post-recognition death toll, multiplying the per-zone brake.
    - ZONE heterogeneity (`zone_r_mult`): a (Z,) multiplier on r_eff standing for
      conflict / accessibility differences between health zones.
    """
    if RECORD_TRACE_DIAG:
        trace_diag_reset()
    z_index = {z: i for i, z in enumerate(zones)}
    if params.seed_zone not in z_index:
        raise ValueError(f"seed zone {params.seed_zone!r} not in zone set")
    if collect_high_from_day is None:
        raise ValueError(
            "collect_high_from_day is required: (Z,) day each zone becomes lab-served"
        )
    seed_zi = z_index[params.seed_zone]
    Z = len(zones)
    # care_destination array. Default = identity (no relocation for anyone).
    if care_destination is None:
        care_destination = np.arange(Z, dtype=np.int64)
    else:
        care_destination = np.asarray(care_destination, dtype=np.int64)

    # current generation state
    cur_zone = np.full(params.seed_n, seed_zi, dtype=np.int64)
    cur_tinf = np.full(params.seed_n, float(params.seed_day))
    cur_trace_cut = np.full(params.seed_n, np.inf)  # traced-contact absolute iso day

    # accumulators
    rec_zone, rec_tinf = [], []
    rec_incub, rec_report, rec_died, rec_otd = [], [], [], []
    rec_collected, rec_confirm = [], []
    rec_isolated, rec_isostart, rec_isoend = [], [], []
    rec_referred, rec_care_zone, rec_t_collect_abs = [], [], []  # referral tracking

    # (Z,) per-zone multiplier on r_eff (conflict / accessibility heterogeneity)
    if zone_r_mult is None:
        zone_mult = np.ones(Z, dtype=float)
    else:
        zone_mult = np.asarray(zone_r_mult, dtype=float)
        if zone_mult.shape != (Z,):
            raise ValueError(f"zone_r_mult must have shape ({Z},), got {zone_mult.shape}")

    # zone-level state for brake + awareness
    tinf_post_by_zone = [
        np.empty(0, dtype=float) for _ in range(Z)
    ]  # per-zone sorted post-recog t_infect (Hill brake driver)
    tinf_post_nat = np.empty(0, dtype=float)  # NATIONAL sorted post-recog t_infect
    conf_abs_nat = np.empty(
        0, dtype=float
    )  # NATIONAL sorted confirmation days (tracing-load driver)
    aware_day = np.full(Z, np.inf)  # first-local-confirmation day per zone
    zone_conf_count = np.zeros(
        Z, dtype=int
    )  # confirmations count per zone (for aware_threshold > 1)

    total = 0
    gen = 0  # generation index; gen == 1 is the seed pass
    while cur_zone.size and total < max_infections:
        gen += 1
        n = cur_zone.size
        rec_zone.append(cur_zone)
        rec_tinf.append(cur_tinf)
        total += n
        if RECORD_TRACE_DIAG:
            _td_listed = np.isfinite(cur_trace_cut)
            TRACE_DIAG["n_inf"] += n
            TRACE_DIAG["n_listed"] += int(_td_listed.sum())
            _td_week(cur_tinf, np.ones(n, bool), 0)
            _td_week(cur_tinf, _td_listed, 1)

        # ---- Hill brake (per-infection r_eff) ----
        # `_decayed_driver` is the cumulative COUNT when tau_w is infinite and the
        # exp-decayed weight sum when it is finite. The generation loop updates the
        # running registries only after a generation completes, so this is a
        # generation-complete online driver, not a fully calendar-complete
        # retrospective death tally.
        cum_for_parent = np.zeros(n, dtype=float)
        threshold_t = cur_tinf - params.brake_lag_days
        for z in np.unique(cur_zone):
            ts = tinf_post_by_zone[z]
            if ts.size > 0:
                zsel = cur_zone == z
                cum_for_parent[zsel] = _decayed_driver(
                    ts, threshold_t[zsel], params.tau_w, params.response_start_day
                )
        if RECORD_BRAKE_DIAG:
            BRAKE_DIAG["rows"].append(
                np.column_stack([cur_tinf, cur_zone.astype(float), cum_for_parent])
            )
        deaths = params.ifr * cum_for_parent
        if params.cum_half > 0:
            ratio = deaths / params.cum_half
            brake = 1.0 / (1.0 + np.power(ratio, params.hill_n))
        else:
            brake = np.where(deaths > 0, 0.0, 1.0)

        # ---- NATIONAL awareness brake (multiplies the local brake) ----
        # Same visible-death units and same lag as the local driver, but counted over
        # the whole country: a zone invaded late is braked by what the country has
        # already seen, not by its own (still empty) toll.
        if params.nat_brake_driver == "calendar":
            if params.nat_half_days > 0:
                elapsed = np.maximum(threshold_t - params.response_start_day, 0.0)
                nat_brake = 1.0 / (
                    1.0 + np.power(elapsed / params.nat_half_days, params.nat_hill_n)
                )
            else:
                nat_brake = np.ones(n)
        elif params.nat_half > 0:
            nat_cum = _decayed_driver(
                tinf_post_nat, threshold_t, params.tau_w, params.response_start_day
            )
            nat_deaths = params.ifr * nat_cum
            nat_brake = 1.0 / (1.0 + np.power(nat_deaths / params.nat_half, params.nat_hill_n))
        else:
            nat_brake = np.ones(n)

        # ---- floor the NATIONAL factor only ----
        # B_nat' = beta + (1-beta)*B_nat. Applied after whichever driver produced nat_brake,
        # so it floors the "calendar" variant identically; a no-op when the national brake is
        # off (floor + (1-floor)*1 == 1). The `> 0` guard keeps beta = 0 bit-identical
        # rather than relying on 0 + 1*x round-tripping exactly.
        if params.nat_floor > 0.0:
            nat_brake = params.nat_floor + (1.0 - params.nat_floor) * nat_brake

        # ---- where the national factor multiplies ----
        # "global" puts it OUTSIDE the bracket, so it scales the braked core and the naive
        # frontier alike; `nat_brake` is 1.0 there by __post_init__ (nat_half/nat_floor forced
        # to 0), so the two forms cannot be layered by accident.
        if params.nat_brake_placement == "global":
            r_eff = nat_scale(threshold_t, params) * (
                params.R_floor + (params.R0 - params.R_floor) * brake
            )
        else:
            r_eff = params.R_floor + (params.R0 - params.R_floor) * brake * nat_brake
        # No brake before recognition
        r_eff[cur_tinf < params.response_start_day] = params.R0
        # zone heterogeneity scales the whole r_eff (floor included) — a
        # better-served, lower-conflict zone transmits less both before and after braking.
        r_eff = r_eff * zone_mult[cur_zone]
        p_nb = params.k / (params.k + r_eff)

        # Update the per-zone post-recog t_infect running set (Hill brake driver)
        post_mask = cur_tinf >= params.response_start_day
        if post_mask.any():
            for z in np.unique(cur_zone[post_mask]):
                zmask = (cur_zone == z) & post_mask
                new_ts = cur_tinf[zmask]
                tinf_post_by_zone[z] = (
                    np.sort(new_ts)
                    if tinf_post_by_zone[z].size == 0
                    else np.sort(np.concatenate([tinf_post_by_zone[z], new_ts]))
                )
            # mirror the national driver (one sorted array over all zones)
            tinf_post_nat = np.sort(np.concatenate([tinf_post_nat, cur_tinf[post_mask]]))

        # ---- parent illness timeline + turnaround ----
        incub = rng.lognormal(params.incub_meanlog, params.incub_sdlog, n)
        # Legacy draw retained to preserve the calibrated RNG stream and tree
        # replay; fixed turnarounds below now determine confirmation timing.
        report = rng.exponential(params.report_delay_mean, n)
        onset_day = cur_tinf + incub
        served = served_from_day[cur_zone]
        local = onset_day >= served  # served zones = local turnaround
        # Time-varying unserved turnaround: pre-switch = Kinshasa (~8d), post-switch = regional hub (~5d)
        unserved_tt = np.where(
            cur_tinf < params.unserved_switch_day,
            params.unserved_turnaround_early,
            params.unserved_turnaround_late,
        )
        turnaround = np.where(local, params.local_turnaround, unserved_tt)

        # ---- tri-state PCR gating ----
        machine_high = onset_day >= collect_high_from_day[cur_zone]
        aware_only = (onset_day >= aware_day[cur_zone]) & ~machine_high
        u_pcr = rng.random(n)  # PCR draw (upstream of nuanced tail)
        died_p = rng.random(n) < params.ifr
        otd_p = rng.gamma(params.otd_shape, params.otd_scale, n)
        rsh = (params.recovery_mean / params.recovery_sd) ** 2
        rsc = params.recovery_sd**2 / params.recovery_mean
        trec_p = rng.gamma(rsh, rsc, n)
        rec_incub.append(incub)
        rec_report.append(report)
        rec_died.append(died_p)
        rec_otd.append(otd_p)
        u = {"pcr": u_pcr, **_draw_nuanced_u_tail(rng, n, params)}
        pcr_prob = np.where(
            machine_high,
            params.p_pcr_high,
            np.where(aware_only, params.p_pcr_mid, params.p_pcr_low),
        )

        iso_cutoff, safe_burial, t_confirm, got_pcr, _t_presump_eff, end_illness, t_collect_rel = (
            _nuanced_iso_burial(
                params,
                n,
                incub,
                died_p,
                otd_p,
                trec_p,
                cur_tinf,
                turnaround,
                pcr_prob,
                machine_high,
                local,
                u,
            )
        )

        # SQ tracing inherited from parent: cur_trace_cut is absolute; fold into iso_cutoff (relative)
        iso_cutoff_notrace = iso_cutoff.copy() if RECORD_TRACE_DIAG else None
        iso_cutoff = np.minimum(iso_cutoff, cur_trace_cut - cur_tinf)
        confirm_abs = cur_tinf + t_confirm
        rec_collected.append(got_pcr)

        # ---- care-seeking referral ----
        # A case is REFERRED if:
        #   - it's in a non-CTE zone (care_destination[zone] != zone)
        #   - its referral draw u["refer"] < f_refer
        # For referred cases: care_zone = destination CTE; post-collect offspring generated there.
        referring_zone = care_destination[cur_zone] != cur_zone
        referred_flag = referring_zone & (u["refer"] < params.f_refer)
        # care_zone for observation (referred cases confirm at CTE)
        care_zone = np.where(referred_flag, care_destination[cur_zone], cur_zone)
        rec_referred.append(referred_flag)
        rec_care_zone.append(care_zone)
        # Absolute t_collect (used for offspring relocation gate)
        t_collect_abs = cur_tinf + t_collect_rel
        rec_t_collect_abs.append(t_collect_abs)

        # ---- nosocomial R excess for referred parents ----
        # brake-independent excess: adds noso_boost * R0 to r_eff for referred cases.
        # Recompute p_nb using the updated r_eff. Only affects offspring counts, not brake driver.
        if params.noso_boost > 0.0:
            r_eff = np.where(referred_flag, r_eff + params.noso_boost * params.R0, r_eff)
            p_nb = params.k / (params.k + r_eff)
        rec_confirm.append(confirm_abs)

        # Isolation census: admission = iso_cutoff (any channel); discharge = min(bed_cap, end_illness)
        bed_cap_rel = np.where(
            got_pcr,
            t_confirm + params.bed_stay_post_confirm_days,
            iso_cutoff + params.bed_stay_suspect_days,
        )
        iso_end_rel = np.minimum(bed_cap_rel, end_illness)
        rec_isolated.append(np.isfinite(iso_cutoff))
        rec_isostart.append(cur_tinf + iso_cutoff)
        rec_isoend.append(cur_tinf + iso_end_rel)

        # ---- awareness: update aware_day when a zone's Nth confirmation lands ----
        tdet = cur_tinf + t_confirm  # confirmation includes both alive-PCR and cadaver channels
        conf = np.isfinite(tdet) & (tdet >= params.response_start_day)
        # grow the national confirmation set BEFORE the tracing block below, so
        # generation g's listing load includes generations <= g. This is causally available:
        # with GI 15.3 d, generation g+1 confirms about a fortnight later and so does not
        # belong in a trailing 14-day window at generation g's confirm day.
        if conf.any():
            conf_abs_nat = np.sort(np.concatenate([conf_abs_nat, tdet[conf]]))
            if RECORD_TRACE_DIAG:
                TRACE_DIAG["conf_days"].append(tdet[conf].copy())  # A2: completed-sim load
        if conf.any():
            if params.aware_threshold <= 1:
                np.minimum.at(aware_day, cur_zone[conf], tdet[conf])
            else:
                cz = cur_zone[conf]
                ct = tdet[conf]
                order = np.argsort(ct)
                for i in order:
                    z = int(cz[i])
                    zone_conf_count[z] += 1
                    if zone_conf_count[z] == params.aware_threshold and aware_day[z] > ct[i]:
                        aware_day[z] = float(ct[i])

        if RECORD_TRACE_DIAG:
            TRACE_DIAG["n_conf"] += int(conf.sum())
            TRACE_DIAG["n_conf_listed"] += int((conf & _td_listed).sum())
            _td_week(cur_tinf, conf, 4)
            _td_week(cur_tinf, conf & _td_listed, 5)

        # ---- offspring ----
        n_off = rng.negative_binomial(params.k, p_nb)
        # Seed conditioning. `gen == 1` names the seed pass directly. The stock
        # draw above still runs so that with the rule off the stream is
        # untouched.
        if params.seed_u_min is not None and gen == 1:
            u_lo = (
                np.power(p_nb, params.k)
                if params.seed_u_min == "nonzero"
                else np.full(n, float(params.seed_u_min))
            )
            uu = u_lo + (1.0 - u_lo) * rng.random(n)
            n_off = _nbinom.ppf(uu, params.k, p_nb).astype(n_off.dtype)
        total_off = int(n_off.sum())
        if total_off == 0:
            break
        parent_idx = np.repeat(np.arange(n), n_off)

        # Ramp-kernel GI: live tail with power exponent + burial pulse (fatal only)
        t_end = np.where(died_p, otd_p, trec_p)[parent_idx]
        live_s = t_end * rng.random(total_off) ** (1.0 / (params.ramp_a + 1.0))
        is_bur = died_p[parent_idx] & (rng.random(total_off) < params.w_burial)
        bur_s = otd_p[parent_idx] + rng.random(total_off) * params.burial_width
        gi = incub[parent_idx] + np.where(is_bur, bur_s, live_s)

        # Averted: parent isolated + offspring past cutoff, OR safe-buried parent's burial-window offspring
        keep = np.where(is_bur, ~safe_burial[parent_idx], gi < iso_cutoff[parent_idx])
        if RECORD_ISO_DIAG:
            ISO_DIAG["case"].append(
                np.column_stack([cur_tinf, iso_cutoff, end_illness, n_off.astype(float)])
            )
            ISO_DIAG["off"].append(
                np.column_stack(
                    [cur_tinf[parent_idx], gi, keep.astype(float), is_bur.astype(float)]
                )
            )
        if RECORD_TRACE_DIAG:
            # what the SAME offspring set would have done with the trace channel removed.
            # Within one run, so all randomness is shared -- unlike a sq_trace_eff = 0 rerun,
            # which desynchronises the RNG stream and is NOT a valid paired counterfactual.
            _keep_nt = np.where(
                is_bur, ~safe_burial[parent_idx], gi < iso_cutoff_notrace[parent_idx]
            )
            _wb_t = cur_tinf[parent_idx] + gi
            TRACE_DIAG["would_be"] += int(_keep_nt.sum())
            TRACE_DIAG["averted_trace"] += int((_keep_nt & ~keep).sum())
            _td_week(_wb_t, _keep_nt, 2)
            _td_week(_wb_t, _keep_nt & ~keep, 3)
        parent_idx = parent_idx[keep]
        gi = gi[keep]
        if parent_idx.size == 0:
            break

        off_tinf = cur_tinf[parent_idx] + gi

        # dispersal
        parent_zone = cur_zone[parent_idx]
        jump = rng.random(parent_idx.size) < params.jump_prob
        off_zone = parent_zone.copy()
        if jump.any():
            ji = np.nonzero(jump)[0]
            for z in np.unique(parent_zone[ji]):
                sel = ji[parent_zone[ji] == z]
                w = mobility_out[z]
                if w.sum() <= 0:
                    off_zone[sel] = z
                else:
                    off_zone[sel] = rng.choice(Z, size=sel.size, p=w)

        # for referred parents, post-collect offspring RELOCATE to CTE.
        # (Physical relocation of the case at t_collect drives subsequent transmission to CTE.)
        if params.f_refer > 0.0:
            parent_referred = referred_flag[parent_idx]
            parent_t_col = t_collect_rel[parent_idx]
            born_post_collect = gi > parent_t_col
            relocated = parent_referred & born_post_collect
            if relocated.any():
                off_zone[relocated] = care_destination[parent_zone[relocated]]

        # SQ tracing: contact of a confirmed parent is traced+isolated at index_confirm + trace_lag, w.p. trace_eff
        if params.sq_trace_eff > 0.0:
            par_conf = confirm_abs[parent_idx]
            # listing-throughput cap. Coverage is discounted by how congested the listing
            # queue was on the PARENT's confirmation day. trace_cap = inf skips this
            # entirely, leaving the draw below bit-identical.
            eff = params.sq_trace_eff
            if np.isfinite(params.trace_cap) or RECORD_TRACE_DIAG:
                if listing_load is not None:
                    # EXOGENOUS load. The response's listing queue was diluted by the caseload
                    # it ACTUALLY faced, which is observed; driving it off the model's own
                    # confirmations cannot work here, because the model's load has the wrong
                    # SHAPE (observed L14 rises 2.83x from mid-June to the horizon while the
                    # model's rises 1.37x and FALLS over the last fortnight, so its coverage
                    # multiplier turns back up exactly when the collapse should happen). Same idiom as the other exogenous response schedules already in
                    # this model: served_from_day, collect_high_from_day, confirm_capacity.
                    # Beyond the array the last value is HELD (documented: for the fit the array
                    # ends at the horizon, so this only affects forward projections).
                    _ld, _lv = listing_load
                    _i = np.clip(
                        np.searchsorted(
                            _ld, np.where(np.isfinite(par_conf), par_conf, _ld[-1]), side="right"
                        )
                        - 1,
                        0,
                        _lv.size - 1,
                    )
                    load = _lv[_i]
                elif conf_abs_nat.size:
                    hi = np.searchsorted(conf_abs_nat, par_conf, side="right")
                    lo = np.searchsorted(conf_abs_nat, par_conf - params.trace_window, side="left")
                    load = (hi - lo).astype(float)
                else:
                    load = np.zeros(parent_idx.size)
                if np.isfinite(params.trace_cap):
                    eff = params.sq_trace_eff * np.minimum(
                        1.0, params.trace_cap / np.maximum(load, 1.0)
                    )
                if RECORD_TRACE_DIAG:
                    TRACE_DIAG["load"].append(
                        np.column_stack([par_conf, load, np.broadcast_to(eff, par_conf.shape)])
                    )
            traced = np.isfinite(par_conf) & (rng.random(parent_idx.size) < eff)
            off_trace_cut = np.where(traced, par_conf + params.sq_trace_lag, np.inf)
        else:
            off_trace_cut = np.full(parent_idx.size, np.inf)

        # keep only offspring infected on or before horizon
        within = off_tinf <= horizon_day
        cur_zone = off_zone[within]
        cur_tinf = off_tinf[within]
        cur_trace_cut = off_trace_cut[within]

    if not rec_zone:
        return pd.DataFrame(
            columns=[
                "zone_idx",
                "care_zone",
                "referred",
                "t_infect",
                "onset",
                "t_detect",
                "detected",
                "died",
                "t_death",
                "isolated",
                "iso_start",
                "iso_end",
                "hit_cap",
            ]
        )

    # ---- observation layer: assemble per-infection DataFrame ----
    zone_idx = np.concatenate(rec_zone)
    t_infect = np.concatenate(rec_tinf)
    n = zone_idx.size
    incub = np.concatenate(rec_incub)
    report = np.concatenate(rec_report)
    died = np.concatenate(rec_died)
    otd = np.concatenate(rec_otd)
    onset = t_infect + incub
    t_death = onset + otd
    detected = np.concatenate(rec_collected)  # got_pcr flag (living detection)
    t_arrive = np.concatenate(rec_confirm)  # confirmation time (absolute) — includes cadaver

    # NB: `detected` currently tracks living-case PCR only (rec_collected).
    # Cadaver-confirmed cases have finite t_arrive but detected = False.
    # For the observation layer we want "any confirmation" = any finite t_arrive.
    detected = np.isfinite(t_arrive)

    # Capacity-limited confirmation FIFO (backlog layer)
    queue_at_query = None
    if confirm_capacity is not None:
        t_detect = np.full(t_arrive.size, np.inf)
        sel = detected & np.isfinite(t_arrive)
        if queue_query_days is not None:
            t_detect[sel], queue_at_query = _confirm_with_capacity(
                t_arrive[sel],
                confirm_capacity[0],
                confirm_capacity[1],
                queue_query_days=queue_query_days,
            )
        else:
            t_detect[sel] = _confirm_with_capacity(
                t_arrive[sel], confirm_capacity[0], confirm_capacity[1]
            )
    else:
        t_detect = t_arrive

    # Isolation census columns
    isolated = np.concatenate(rec_isolated)
    iso_start = np.concatenate(rec_isostart)
    iso_end = np.concatenate(rec_isoend)

    # care_zone for referred cases (obs-side reassignment to CTE)
    referred_arr = np.concatenate(rec_referred) if rec_referred else np.zeros(n, dtype=bool)
    care_zone = np.concatenate(rec_care_zone) if rec_care_zone else zone_idx.copy()

    df = pd.DataFrame(
        {
            "zone_idx": zone_idx,
            "care_zone": care_zone,  # CTE zone for referred cases; = zone_idx otherwise
            "referred": referred_arr,  # was this case referred to a CTE?
            "t_infect": t_infect,
            "onset": onset,
            "t_detect": t_detect,
            "detected": detected,
            "died": died,
            "t_death": t_death,
            "isolated": isolated,
            "iso_start": iso_start,
            "iso_end": iso_end,
            "hit_cap": total >= max_infections,
        }
    )
    if queue_query_days is not None:
        df.attrs["queue_at_query"] = (
            queue_at_query if queue_at_query is not None else np.zeros(len(queue_query_days))
        )
        df.attrs["queue_query_days"] = np.asarray(queue_query_days)
    return df
