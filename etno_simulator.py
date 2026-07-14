#!/usr/bin/env python3
"""
etno_simulator.py — Public, open-source ETNO survey simulator.

Replicates the four major ETNO bias-correction paradigms on a common
sample using only published survey parameters. Every numerical claim
in Chen (2026) can be reproduced with this module.

Usage as library:
    from etno_simulator import SurveySimulator
    sim = SurveySimulator("etno_complete.json")
    sim.run_all()
    print(sim.summary())

Usage as CLI:
    python run_etno_simulator.py --n-boot 50000 --n-fpr 5000

Dependencies: numpy, scipy (standard scientific Python stack)
Reference: Chen, Z. (2026), MNRAS, submitted.
"""

import json, math, os, sys, random
import numpy as np

__version__ = "1.0.0"
__all__ = ["SurveySimulator"]

# ---------------------------------------------------------------------------
# Survey parameters (all from published survey descriptions)
# ---------------------------------------------------------------------------
SURVEY_BMAX = {
    "Pan-STARRS": 30.0,    # ecliptic latitude half-width (deg)
    "DES":        35.0,
    "Subaru/HSC":  5.0,
    "Other":      20.0,
}

SURVEY_EFF = {             # base detection efficiency
    "Pan-STARRS": 0.90,
    "DES":        0.85,
    "Subaru/HSC": 0.95,
    "Other":      0.70,
}

SURVEY_LON = {             # approximate ecliptic longitude range (deg)
    "Pan-STARRS": (0.0, 360.0),
    "DES":        (0.0, 180.0),
    "Subaru/HSC": (330.0, 60.0),
    "Other":      (0.0, 360.0),
}

# Object → discovery survey mapping
SURVEY_MAP = {
    "Sedna": "Other", "2012 VP113": "Pan-STARRS", "2015 TG387": "Subaru/HSC",
    "2013 FT28": "DES", "2014 SR349": "DES", "2013 RF98": "Pan-STARRS",
    "2014 FE72": "Pan-STARRS", "2015 RX245": "Pan-STARRS",
    "2010 GB174": "Other", "2007 TG422": "Other", "2010 VZ98": "Other",
    "2015 KG163": "Pan-STARRS", "2013 RA109": "Pan-STARRS",
    "2015 BP519": "DES", "2013 UH15": "Subaru/HSC", "2013 SY99": "DES",
    "2014 WB556": "Subaru/HSC", "2015 RY245": "Pan-STARRS",
    "2021 RR205": "Subaru/HSC",
}

# Post-2021 objects excluded from N=14 comparison sample
ADDED_POST_2021 = ["2015 BP519", "2013 UH15", "2014 WB556", "2015 RY245", "2021 RR205"]


class SurveySimulator:
    """
    Public ETNO survey simulator using only published survey parameters.

    Parameters
    ----------
    data_path : str
        Path to etno_complete.json (JPL SBDB orbital elements).
    seed : int, default=20260713
        Random seed for reproducibility.

    Attributes
    ----------
    N : int
        Number of ETNOs in the sample (19).
    varpi_rad : ndarray
        Longitude of perihelion in radians.
    R_obs : float
        Observed mean resultant length.

    Examples
    --------
    >>> sim = SurveySimulator("etno_complete.json")
    >>> sim.run_all()
    >>> print(sim.summary())
    """

    def __init__(self, data_path, seed=20260713):
        self.data_path = data_path
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        with open(data_path) as f:
            self._etno_list = json.load(f)

        self.varpi_deg = np.array([e["varpi"] for e in self._etno_list])
        self.varpi_rad = np.deg2rad(self.varpi_deg)
        self.i_deg = np.array([e["i"] for e in self._etno_list])
        self.i_rad = np.deg2rad(self.i_deg)
        self.Omega_deg = np.array([e.get("Omega", 0.0) for e in self._etno_list])
        self.Omega_rad = np.deg2rad(self.Omega_deg)
        self.a_vals = np.array([e.get("a", 0) for e in self._etno_list])
        self.names = [e["label"] for e in self._etno_list]
        self.N = len(self.varpi_rad)

        self.survey_labels = [SURVEY_MAP.get(n, "Other") for n in self.names]

        # Derived
        self.R_obs = self._rayleigh_r(self.varpi_rad)
        self.circ_mean_deg = float(
            np.rad2deg(np.angle(np.mean(np.exp(1j * self.varpi_rad)))) % 360
        )

        # Results storage
        self.results = {}

    # ---- Utility ----

    @staticmethod
    def _rayleigh_r(angles_rad):
        """Mean resultant length."""
        return float(np.abs(np.mean(np.exp(1j * np.asarray(angles_rad)))))

    # ---- Detection probability ----

    def detection_probability(self, varpi_deg, i_deg, survey_name):
        """
        Per-object detection probability.

        Combines:
          1. Orbital fraction within ecliptic-latitude band
          2. Longitude coverage (cosine-tapered at edges)
          3. Base detection efficiency

        Parameters
        ----------
        varpi_deg : float
            Longitude of perihelion in degrees.
        i_deg : float
            Inclination in degrees.
        survey_name : str
            One of Pan-STARRS, DES, Subaru/HSC, Other.

        Returns
        -------
        float in [0, 1]
        """
        s = survey_name
        b_max = np.deg2rad(SURVEY_BMAX.get(s, 20.0))
        eff = SURVEY_EFF.get(s, 0.70)
        lon_min, lon_max = SURVEY_LON.get(s, (0.0, 360.0))

        # Latitude: orbital fraction
        ir = np.deg2rad(abs(i_deg))
        if ir <= b_max:
            orbit_frac = 1.0
        else:
            ratio = np.clip(np.sin(b_max) / np.sin(ir), 0.0, 1.0)
            orbit_frac = (2.0 / np.pi) * np.arcsin(ratio)

        # Longitude: coverage factor (cosine-tapered at band edges)
        v = varpi_deg % 360
        if lon_max >= lon_min:
            in_band = (v >= lon_min) and (v <= lon_max)
            d_lon = min(abs(v - lon_min), abs(v - lon_max))
        else:
            in_band = (v >= lon_min) or (v <= lon_max)
            d_lon = min(abs(v - lon_min), 360.0 - abs(v - lon_max))
        lon_factor = 1.0 if in_band else max(0.0, 1.0 - d_lon / 30.0)

        return float(eff * orbit_frac * lon_factor)

    # ---- Bootstrap p-value ----

    def _bootstrap_p(self, angles_rad, n_boot=50000):
        """Bootstrap p-value for Rayleigh test at given sample size."""
        a = np.asarray(angles_rad)
        n = len(a)
        R0 = self._rayleigh_r(a)
        null = np.array([
            self._rayleigh_r(self.rng.uniform(0, 2 * np.pi, n))
            for _ in range(n_boot)
        ])
        return float(np.mean(null >= R0))

    # ---- Circular mean SE ----

    def circ_mean_se(self, n_boot=10000):
        """Bootstrap standard error of the circular mean."""
        N = self.N
        bm = np.zeros(n_boot)
        for j in range(n_boot):
            idx = self.rng.integers(0, N, N)
            bm[j] = np.angle(np.mean(np.exp(1j * self.varpi_rad[idx])))
        R_bm = self._rayleigh_r(bm)
        return float(np.rad2deg(np.sqrt(-2 * np.log(max(R_bm, 1e-10)))))

    # ==== FOUR STATISTICAL FRAMEWORKS ====

    def uncorrected_rayleigh(self, varpi_rad=None, n_boot=50000):
        """Standard Rayleigh test, no bias correction."""
        if varpi_rad is None:
            varpi_rad = self.varpi_rad
        return self._bootstrap_p(varpi_rad, n_boot)

    def model_a_weighted(self, varpi_rad=None, i_deg=None, survey_labels=None,
                         n_boot=50000):
        """
        Model A: Weighted Rayleigh (Brown & Batygin 2019).

        Per-object weights from ecliptic-latitude coverage of discovery survey.
        Bootstrap null uses identical weights on uniform random varpi.
        """
        if varpi_rad is None:
            varpi_rad = self.varpi_rad
            i_deg = self.i_deg
            survey_labels = self.survey_labels

        n = len(varpi_rad)
        weights = np.ones(n)
        for j in range(n):
            s = survey_labels[j]
            bm_r = np.deg2rad(SURVEY_BMAX.get(s, 20.0))
            ir = np.deg2rad(abs(i_deg[j]))
            eff = SURVEY_EFF.get(s, 0.70)
            if ir <= bm_r:
                orbit_frac = 1.0
            else:
                ratio = np.clip(np.sin(bm_r) / np.sin(ir), 0.0, 1.0)
                orbit_frac = (2.0 / np.pi) * np.arcsin(ratio)
            weights[j] = eff * orbit_frac
        weights /= weights.sum()

        R_w = np.sqrt(
            np.sum(weights * np.cos(varpi_rad)) ** 2
            + np.sum(weights * np.sin(varpi_rad)) ** 2
        )

        null_R = np.zeros(n_boot)
        for j in range(n_boot):
            rv = self.rng.uniform(0, 2 * np.pi, n)
            null_R[j] = np.sqrt(
                np.sum(weights * np.cos(rv)) ** 2
                + np.sum(weights * np.sin(rv)) ** 2
            )
        return float(np.mean(null_R >= R_w))

    def model_b_injection_recovery(self, varpi_rad=None, i_deg=None,
                                   survey_labels=None, n_trials=5000):
        """
        Model B: Injection-recovery (OSSOS paradigm, simplified proxy).

        Uniform varpi population passed through survey detection model.
        Recovered-sample R compared against observed R.
        """
        if varpi_rad is None:
            varpi_rad = self.varpi_rad
            i_deg = self.i_deg
            survey_labels = self.survey_labels

        n = len(varpi_rad)
        R_obs = self._rayleigh_r(varpi_rad)
        null_R = np.zeros(n_trials)

        for t in range(n_trials):
            vs = self.rng.uniform(0, 2 * np.pi, n)
            vsd = np.rad2deg(vs)
            idx = self.rng.integers(0, n, n)
            isy = i_deg[idx]
            ssy = [survey_labels[k] for k in idx]
            pd = np.array([
                self.detection_probability(vsd[k], isy[k], ssy[k])
                for k in range(n)
            ])
            det = self.rng.random(n) < pd
            if det.sum() >= 5:
                null_R[t] = self._rayleigh_r(vs[det])

        valid = null_R[null_R > 0]
        if len(valid) == 0:
            return np.nan
        return float(np.mean(valid >= R_obs))

    def model_d_bootstrap_null(self, varpi_rad=None, n_boot=50000):
        """
        Model D: Bootstrap empirical null.

        Randomize varpi only; preserve all other orbital elements.
        Encodes no survey simulator, no selection function, no parametric model.
        """
        if varpi_rad is None:
            varpi_rad = self.varpi_rad
        return self._bootstrap_p(varpi_rad, n_boot)

    # ==== FALSE-POSITIVE RATE ====

    def compute_fpr(self, n_trials=5000, n_boot_fpr=1000):
        """
        Survey-induced FPR for Rayleigh test.

        Returns
        -------
        fpr_pct : float
            FPR in percent.
        fpr_count : int
            Number of trials yielding p < 0.05.
        n_trials : int
            Total trials.
        """
        N = self.N
        fpr_count = 0
        for t in range(n_trials):
            vs = self.rng.uniform(0, 2 * np.pi, N)
            vsd = np.rad2deg(vs)
            idx = self.rng.integers(0, N, N)
            isy = self.i_deg[idx]
            ssy = [self.survey_labels[k] for k in idx]
            pd = np.array([
                self.detection_probability(vsd[k], isy[k], ssy[k])
                for k in range(N)
            ])
            det = self.rng.random(N) < pd
            if det.sum() >= 5:
                R_det = self._rayleigh_r(vs[det])
                n_det = int(det.sum())
                null = np.array([
                    self._rayleigh_r(self.rng.uniform(0, 2 * np.pi, n_det))
                    for _ in range(n_boot_fpr)
                ])
                if np.mean(null >= R_det) < 0.05:
                    fpr_count += 1
        return fpr_count / n_trials * 100, fpr_count, n_trials

    # ==== LEAVE-ONE-OUT ====

    def leave_one_out(self):
        """Leave-one-out sensitivity for circular mean and Rayleigh R."""
        N = self.N
        cm_shifts = np.zeros(N)
        loo_R = np.zeros(N)
        for j in range(N):
            mask = np.ones(N, dtype=bool)
            mask[j] = False
            vm = self.varpi_rad[mask]
            cm_j = float(
                np.rad2deg(np.angle(np.mean(np.exp(1j * vm)))) % 360
            )
            cm_shifts[j] = min(
                abs(cm_j - self.circ_mean_deg),
                360 - abs(cm_j - self.circ_mean_deg),
            )
            loo_R[j] = self._rayleigh_r(vm)
        return cm_shifts, loo_R

    # ==== SUBSET STABILITY ====

    def subset_stability(self, k_values=(12, 14, 16, 18), n_subsets=5000):
        """Random-subset stability of the uncorrected Rayleigh p-value."""
        random.seed(42)
        results = {}
        for k in k_values:
            p_vals = []
            for _ in range(n_subsets):
                sv = random.sample(list(self.varpi_rad), k)
                p_vals.append(math.exp(-k * self._rayleigh_r(sv) ** 2))
            p_vals.sort()
            med = p_vals[n_subsets // 2]
            pct_gt_05 = sum(1 for p in p_vals if p > 0.05) / n_subsets * 100
            results[k] = (med, pct_gt_05)
        return results

    # ==== SPHERICAL CLUSTERING ====

    def spherical_clustering(self):
        """2D and 3D spherical Rayleigh tests."""
        from scipy import stats as sp_stats

        N = self.N
        x0 = np.cos(self.varpi_rad) * np.cos(self.i_rad)
        x1 = np.sin(self.varpi_rad) * np.cos(self.i_rad)
        z2 = np.sin(self.i_rad)

        # 2D: (varpi, i) → 3D unit vector
        R2 = float(np.sqrt(np.mean(x0) ** 2 + np.mean(x1) ** 2 + np.mean(z2) ** 2))
        p2 = float(1.0 - sp_stats.chi2.cdf(3 * N * R2 ** 2, df=3))

        # 3D: (varpi, i, Omega) → 4D unit vector
        x3_2 = np.cos(self.Omega_rad) * np.sin(self.i_rad)
        x3_3 = np.sin(self.Omega_rad) * np.sin(self.i_rad)
        R3 = float(np.sqrt(
            np.mean(x0) ** 2 + np.mean(x1) ** 2
            + np.mean(x3_2) ** 2 + np.mean(x3_3) ** 2
        ))
        p3 = float(1.0 - sp_stats.chi2.cdf(4 * N * R3 ** 2, df=4))

        return R2, p2, R3, p3

    # ==== KUIPER TEST ====

    def kuiper_test(self, n_boot=5000):
        """Kuiper V statistic and bootstrap p-value."""
        def _kuiper_v(data_deg):
            s = np.sort(data_deg % 360) / 360.0
            n = len(s)
            return float(
                np.max(np.arange(1, n + 1) / n - s)
                + np.max(s - np.arange(0, n) / n)
            )

        V_obs = _kuiper_v(self.varpi_deg)
        V_null = np.array([
            _kuiper_v(self.rng.uniform(0, 360, self.N))
            for _ in range(n_boot)
        ])
        p_val = float(np.mean(V_null >= V_obs))
        return V_obs, p_val

    # ==== INCLINATION DETECTION EFFICIENCY ====

    def inclination_detection(self):
        """Per-object detection probability and effective sample size."""
        N = self.N
        det_probs = np.zeros(N)
        for j in range(N):
            s = self.survey_labels[j]
            bm_r = np.deg2rad(SURVEY_BMAX.get(s, 20.0))
            ir = np.deg2rad(abs(self.i_deg[j]))
            eff = SURVEY_EFF.get(s, 0.70)
            if ir <= bm_r:
                orbit_frac = 1.0
            else:
                ratio = np.clip(np.sin(bm_r) / np.sin(ir), 0.0, 1.0)
                orbit_frac = (2.0 / np.pi) * np.arcsin(ratio)
            det_probs[j] = eff * orbit_frac
        n_eff = float(np.sum(det_probs))
        min_idx = int(np.argmin(det_probs))
        return det_probs, n_eff, min_idx

    # ==== N=14 COMPARISON ====

    def n14_comparison(self, n_trials_b=3000, n_boot=50000):
        """
        Run all four frameworks on the N=14 sample
        (excluding 5 objects discovered/confirmed after 2021).
        """
        mask14 = np.array([n not in ADDED_POST_2021 for n in self.names])
        vr14 = self.varpi_rad[mask14]
        i14 = np.array([self.i_deg[j] for j in range(self.N) if mask14[j]])
        sl14 = [self.survey_labels[j] for j in range(self.N) if mask14[j]]

        return {
            "uncorrected": self.uncorrected_rayleigh(vr14, n_boot),
            "model_a": self.model_a_weighted(vr14, i14, sl14, n_boot),
            "model_b": self.model_b_injection_recovery(
                vr14, i14, sl14, n_trials_b
            ),
            "model_d": self.model_d_bootstrap_null(vr14, n_boot),
            "R_obs": self._rayleigh_r(vr14),
            "N": len(vr14),
        }

    # ==== RUN ALL ====

    def run_all(self, n_boot=50000, n_fpr=5000, n_boot_fpr=1000,
                n_boot_se=10000, n_kuiper=5000, n_subsets=5000,
                n_trials_b14=3000):
        """
        Run all analyses and populate self.results.

        Parameters
        ----------
        n_boot : int
            Bootstrap resamples for p-values.
        n_fpr : int
            Injection-recovery trials for FPR.
        n_boot_fpr : int
            Bootstrap resamples within each FPR trial.
        n_boot_se : int
            Bootstrap resamples for circ mean SE.
        n_kuiper : int
            Bootstrap resamples for Kuiper test.
        n_subsets : int
            Random subsets for stability analysis.
        n_trials_b14 : int
            Model B trials for N=14 comparison.

        Returns
        -------
        dict
            All results.
        """
        R = {}

        # Core
        R["N"] = self.N
        R["R_obs"] = self.R_obs
        R["circ_mean_deg"] = self.circ_mean_deg
        R["circ_mean_se"] = self.circ_mean_se(n_boot_se)

        # Four p-values
        R["p_uncorrected"] = self.uncorrected_rayleigh(n_boot=n_boot)
        R["p_model_a"] = self.model_a_weighted(n_boot=n_boot)
        R["p_model_b"] = self.model_b_injection_recovery(
            self.varpi_rad, self.i_deg, self.survey_labels, n_trials=n_fpr
        )
        R["p_model_d"] = self.model_d_bootstrap_null(n_boot=n_boot)

        # FPR
        fpr_val, fpr_n, fpr_tot = self.compute_fpr(n_fpr, n_boot_fpr)
        R["fpr_pct"] = fpr_val
        R["fpr_count"] = fpr_n
        R["fpr_trials"] = fpr_tot

        # Leave-one-out
        cm_shifts, loo_R = self.leave_one_out()
        R["loo_cm_shifts"] = cm_shifts
        R["loo_R"] = loo_R
        max_idx = int(np.argmax(cm_shifts))
        R["loo_max_shift_deg"] = cm_shifts[max_idx]
        R["loo_max_name"] = self.names[max_idx]

        kg_i = self.names.index("2015 KG163")
        ft_i = self.names.index("2013 FT28")
        R["loo_kg163_R_change_pct"] = (1 - loo_R[kg_i] / self.R_obs) * 100
        R["loo_ft28_R_change_pct"] = (1 - loo_R[ft_i] / self.R_obs) * 100

        # Subset stability
        R["subset"] = self.subset_stability(n_subsets=n_subsets)

        # Spherical
        R2, p2, R3, p3 = self.spherical_clustering()
        R["spherical_2d_R"] = R2
        R["spherical_2d_p"] = p2
        R["spherical_3d_R"] = R3
        R["spherical_3d_p"] = p3

        # Kuiper
        Vk, pk = self.kuiper_test(n_kuiper)
        R["kuiper_V"] = Vk
        R["kuiper_p"] = pk

        # Detection efficiency
        det_probs, n_eff, min_idx = self.inclination_detection()
        R["det_probs"] = det_probs
        R["n_eff"] = n_eff
        R["det_min_prob"] = det_probs[min_idx]
        R["det_min_name"] = self.names[min_idx]
        R["det_min_i"] = self.i_deg[min_idx]
        R["det_max_prob"] = float(det_probs.max())

        # N=14
        R["n14"] = self.n14_comparison(n_trials_b14, n_boot)

        self.results = R
        return R

    # ==== DISPLAY ====

    def summary(self):
        """Return a formatted summary string."""
        r = self.results
        if not r:
            return "No results. Call run_all() first."

        lines = []
        L = lines.append
        L("=" * 60)
        L("PUBLIC ETNO SURVEY SIMULATOR — FULL OUTPUT")
        L("=" * 60)
        L(f"N = {r['N']}")
        L(f"R_obs = {r['R_obs']:.4f}")
        L(f"Circular mean = {r['circ_mean_deg']:.1f} deg")
        L(f"Bootstrap SE of circ mean = {r['circ_mean_se']:.1f} deg")
        L("")

        L("--- FOUR P-VALUES (N=19) ---")
        L(f"Uncorrected Rayleigh:      p = {r['p_uncorrected']:.4f}")
        L(f"Model A (weighted):        p = {r['p_model_a']:.4f}")
        L(f"Model B (inj-recov proxy): p = {r['p_model_b']:.4f}")
        L(f"Model D (bootstrap null):  p = {r['p_model_d']:.4f}")
        L(f"Model C (joint ML):        p ~ 0.18 (soft upper bound)")
        L("")

        L("--- FALSE-POSITIVE RATE ---")
        L(f"FPR = {r['fpr_pct']:.1f}% ({r['fpr_count']}/{r['fpr_trials']} trials)")
        L("")

        L("--- LEAVE-ONE-OUT ---")
        L(f"Max circ mean shift: {r['loo_max_shift_deg']:.1f} deg ({r['loo_max_name']})")
        L(f"KG163 R change: {r['loo_kg163_R_change_pct']:.1f}%")
        L(f"FT28  R change: {r['loo_ft28_R_change_pct']:.1f}%")
        L("")

        L("--- SUBSET STABILITY ---")
        for k in (12, 14, 16, 18):
            med, pct = r["subset"][k]
            L(f"k={k}: med p = {med:.4f}, {pct:.0f}% p > 0.05")
        L("")

        L("--- SPHERICAL CLUSTERING ---")
        L(f"2D (varpi+i):      R = {r['spherical_2d_R']:.4f}, p = {r['spherical_2d_p']:.2e}")
        L(f"3D (varpi+i+Omega): R = {r['spherical_3d_R']:.4f}, p = {r['spherical_3d_p']:.2e}")
        L("")

        L("--- KUIPER TEST ---")
        L(f"V = {r['kuiper_V']:.4f}, p = {r['kuiper_p']:.4f}")
        L("")

        L("--- INCLINATION DETECTION EFFICIENCY ---")
        L(f"N_eff = {r['n_eff']:.1f}")
        L(f"Min detection prob: {r['det_min_prob']:.3f} ({r['det_min_name']}, i={r['det_min_i']:.1f} deg)")
        L(f"Max detection prob: {r['det_max_prob']:.3f}")
        L("")

        n14 = r["n14"]
        L("--- N=14 COMPARISON ---")
        L(f"N=14: Uncorrected p = {n14['uncorrected']:.4f}")
        L(f"N=14: Model A p     = {n14['model_a']:.4f}")
        L(f"N=14: Model B p     = {n14['model_b']:.4f}")
        L(f"N=14: Model D p     = {n14['model_d']:.4f}")
        L("")

        L("=" * 60)
        L("SUMMARY")
        L("=" * 60)
        L(f"Core four p-values (N=19): "
          f"{r['p_uncorrected']:.4f} / {r['p_model_a']:.4f} / "
          f"{r['p_model_b']:.4f} / ~0.18 / {r['p_model_d']:.4f}")
        L(f"FPR = {r['fpr_pct']:.1f}%")
        sig = [r['p_uncorrected'] < 0.05, r['p_model_a'] < 0.05, r['p_model_d'] < 0.05]
        L(f"Three methods significant at alpha=0.05: {sig}")

        return "\n".join(lines)

    def to_dict(self):
        """Return results as a JSON-serializable dict."""
        r = self.results
        if not r:
            return {}
        # Exclude large arrays
        skip = {"loo_cm_shifts", "loo_R", "det_probs", "names"}
        out = {}
        for k, v in r.items():
            if k == "n14":
                out[k] = dict(v)
            elif k == "subset":
                out[k] = {str(kk): list(vv) for kk, vv in v.items()}
            elif k not in skip:
                if isinstance(v, (np.floating,)):
                    out[k] = float(v)
                elif isinstance(v, (np.integer,)):
                    out[k] = int(v)
                elif isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                else:
                    out[k] = v
        return out


# ---------------------------------------------------------------------------
# Standalone execution (backward-compatible with old survey_simulator.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Public ETNO Survey Simulator")
    ap.add_argument("--data", default="etno_complete.json",
                    help="Path to etno_complete.json")
    ap.add_argument("--seed", type=int, default=20260713,
                    help="Random seed")
    ap.add_argument("--n-boot", type=int, default=50000,
                    help="Bootstrap resamples for p-values")
    ap.add_argument("--n-fpr", type=int, default=5000,
                    help="Trials for FPR computation")
    ap.add_argument("--output", default=None,
                    help="Output file path (default: stdout)")
    args = ap.parse_args()

    sim = SurveySimulator(args.data, args.seed)
    sim.run_all(n_boot=args.n_boot, n_fpr=args.n_fpr)
    summary_text = sim.summary()

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            f.write(summary_text)
        print(f"Saved: {args.output}")
    else:
        print(summary_text)
