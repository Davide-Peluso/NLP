"""
GLM logit per `hard_toxic_neg` – predictors:
    • platform  (0 = Twitch, 1 = YouTube)
    • msg_len z-score
    • language        (β one per language, somma β = 0)

η = intercept
    + β_platform * platform_code
    + β_msglen   * msg_z
    + β_lang[k]  * 1_{lingua=k}
"""

from __future__ import annotations
import os, argparse, random
from typing import Optional

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────── I/O ───────────────────────
def _load_any(path: str) -> pd.DataFrame:
    ext = os.path.splitext(str(path))[1].lower()
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if ext in {".feather", ".ft"}:
        return pd.read_feather(path)
    return pd.read_csv(path, low_memory=False)

# ─────────────────── DATA PREP ─────────────────────
def load_df(path: str) -> pd.DataFrame:
    df = _load_any(path)

    required = {"language", "platform", "msg_len", "hard_toxic_neg"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns: {missing}")

    df[["language", "platform"]] = df[["language", "platform"]].astype("category")

    df["msg_z"] = (
        (df["msg_len"] - df["msg_len"].mean()) / df["msg_len"].std(ddof=0)
    ).astype("float32")

    for c in ["language", "platform"]:
        df[f"{c}_code"] = df[c].cat.codes.astype("int32")

    df["hard_toxic_neg"] = df["hard_toxic_neg"].astype("int8")
    if set(df["hard_toxic_neg"].unique()) - {0, 1}:
        raise ValueError("hard_toxic_neg must be 0/1")

    return df

# ──────────────────── MODEL ───────────────────────
def build_glm(df: pd.DataFrame) -> pm.Model:
    n_l = df["language_code"].nunique()

    y    = df["hard_toxic_neg"].values
    plat = df["platform_code"].values
    lang = df["language_code"].values
    mlen = df["msg_z"].values

    with pm.Model() as m:
        # intercept & fixed effects
        a0     = pm.Normal("intercept", 0, 1)
        b_plat = pm.Normal("beta_platform", 0, 1)
        b_len  = pm.Normal("beta_msglen",   0, 1)

        # un β per lingua con vincolo somma = 0
        b_raw   = pm.Normal("beta_lang_raw", 0, 1, shape=n_l)
        b_lang  = pm.Deterministic(
            "beta_lang",
            b_raw - pm.math.mean(b_raw)      # centra → somma a 0
        )

        eta = a0 + b_plat * plat + b_len * mlen + b_lang[lang]
        pm.Bernoulli("likelihood", logit_p=eta, observed=y)

    return m

# ─────────────────── RUNNER ───────────────────────
def run(
    data_path: str = "ready_all_langs.parquet",
    out_dir: str   = "glm_lang_plat_len",
    *,
    sample_frac: float = 0.05,
    draws: int = 2000,
    tune: int = 2000,
    chains: int = 4,
    cores: int = 4,
    target_accept: float = 0.95,
    prep_only: bool = False,
    resume: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)

    # ---------- load & subsample ----------
    df = load_df(data_path)
    if sample_frac < 1.0:
        df = (
            df.sample(frac=sample_frac, random_state=SEED)
              .reset_index(drop=True)
        )
        for c in ["language", "platform"]:
            df[c] = df[c].cat.remove_unused_categories()
            df[f"{c}_code"] = df[c].cat.codes.astype("int32")

    # ---------- save pre-processed DF ----------
    prep_f = os.path.join(out_dir, "ready_df.parquet")
    df.to_parquet(prep_f)
    print(f"✔ DataFrame salvato → {prep_f}")

    if prep_only:
        print("prep_only=True → stop before sampling.")
        return None

    # ---------- file paths ----------
    trace_f = os.path.join(out_dir, "trace.nc")
    csv_f   = os.path.join(out_dir, "summary.csv")
    json_f  = os.path.join(out_dir, "summary.json")

    # ---------- resume logic ----------
    id_prev = az.from_netcdf(resume) if resume else None
    if id_prev is not None:
        tune = 0  # niente warm-up se riprendiamo

    # ---------- sampling ----------
    with build_glm(df):
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            idata=id_prev,
            random_seed=SEED,
            progressbar=True,
        )

    # ---------- save outputs ----------
    az.to_netcdf(idata, trace_f)
    summ = az.summary(idata, round_to=3)
    summ.to_csv(csv_f)
    summ.to_json(json_f, orient="index", indent=2)

    print(f"✔ Trace   → {trace_f}\n✔ Summary → {csv_f}")
    return summ

# ───────────────────── CLI ───────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="ready_all_langs.parquet")
    ap.add_argument("--out",  default="glm_lang_plat_len")
    ap.add_argument("--frac", type=float, default=0.05)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune",  type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores",  type=int, default=4)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--prep_only", action="store_true")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    run(
        data_path=args.data,
        out_dir=args.out,
        sample_frac=args.frac,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        cores=args.cores,
        target_accept=args.target_accept,
        prep_only=args.prep_only,
        resume=args.resume,
    )
