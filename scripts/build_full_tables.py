#!/usr/bin/env python
"""Build a FULL-design table dir: every design point that exists, no QA / no pT cut baked in.

    usage: myenv312/bin/python scripts/build_full_tables.py <curated_root> <out_tables_dir>
    e.g.   myenv312/bin/python scripts/build_full_tables.py \
             /tmp/curated/jetscape_curated_tables  data/20230320_full/tables
    (extract jetscape_curated_tables.tar.gz first; this script only WRITES into <out_tables_dir>)

Read-only sources:
  Design      STAT/input/STAT20230116Exponential/Design__exponential.dat  (230 rows, LINEAR;
              its first 200 rows == STAT20230320Exponential/Design.dat)
  Prediction  STAT/input/STAT20230320Exponential/Prediction_*_{values,errors}.dat
              (columns labelled design_point<i> in the header; NOT every observable has all 230:
              ids 13, 15, 103 were never computed everywhere -> common set is 227)
  Data bins   STAT/input/STAT20230320Exponential/Data_*.dat, filtered by the paper YAML's DataExclude
              (= the pT > 5 GeV floor JETSCAPE was run with; the extra MinPT-9 cut is NOT applied here)
  Systematics jetscape_curated_tables (named per-source: correlated/uncorrelated/taa/luminosity/...)
              mapped onto those bins

Design-point dropping (QA), pT cuts and smoothing are deliberately NOT applied -- they are config
choices in the Bayesian pipeline (design_points_to_exclude / cuts / preprocessing).

Two hard-won correctness rules encoded below:
  * stat and sys MUST come from the SAME source file. STAT's files and the curated files put the
    same uncertainty in different columns for some tables (STAR-200: stat=0/sys=0.03 vs
    stat=0.03/no-sys); mixing them produced 24 zero-uncertainty bins -> singular covariance.
  * 6 observables have predictions whose binning matches no available data file (declared data
    files don't exist anywhere) -> skipped LOUDLY, never silently.
"""
import os, re, sys, glob
import numpy as np
import yaml

STAT = "/Users/zhanj82/Desktop/JetScapeSTAT"
YAML_F = f"{STAT}/Emulators/yaml/Exponential20230320RBF_N4.yaml"
IN = f"{STAT}/STAT/input/STAT20230320Exponential"
DESIGN_SRC = f"{STAT}/STAT/input/STAT20230116Exponential/Design__exponential.dat"
CUR, OUT = sys.argv[1], sys.argv[2]


def rows(f):
    return np.array([[float(x) for x in l.split()] for l in open(f)
                     if not l.startswith("#") and l.strip()])


def header_cols(f):
    for l in open(f):
        if l.startswith("# Label"):
            return l.split()[2:]
    return []


def dp_labels(f):
    for l in open(f):
        if l.startswith("# design_point"):
            return [int(m) for m in re.findall(r"design_point(\d+)", l)]
    return []


def label_of(name, attr):
    """STAT observable key (e.g. 'ALICE_2760_Hadron_ch_0_5') -> our label
    ('2760__PbPb__hadron__pt_ch_alice____0-5'). Verified 92/92 against the curated tables."""
    exp = name.split("_")[0].lower()
    E = attr["Energy"]; system = "AuAu" if E == 200 else "PbPb"
    o = attr["Observable"]; cent = f'{attr["CentralityMin"]}-{attr["CentralityMax"]}'
    if o.startswith("Hadron"):
        sp = {"HadronCh": "ch", "HadronPi": "pi", "HadronPi0": "pi0"}[o]
        return f"{E}__{system}__hadron__pt_{sp}_{exp}____{cent}"
    m = re.match(r"(Charged)?JetR(\d)(\d)$", o)
    typ = "inclusive_chjet" if m.group(1) else "inclusive_jet"
    return f"{E}__{system}__{typ}__pt_{exp}__R{m.group(2)}.{m.group(3)}__{cent}"


items = yaml.safe_load(open(YAML_F))["Data"]
D = rows(DESIGN_SRC)
assert D.shape == (230, 6), D.shape

# ---- Which design points were actually COMPUTED for every observable? (data fact, not QA)
common = None
for _v in items.values():
    s = set(dp_labels(f"{IN}/{_v['Prediction']}"))
    common = s if common is None else (common & s)
COMMON = sorted(common)
print(f"design points computed for ALL observables: {len(COMMON)}"
      f"   (never computed everywhere: {sorted(set(range(230)) - set(COMMON))})")

for sub in ("Data", "Design", "Prediction"):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)

with open(f"{OUT}/Design/Design__exponential.dat", "w") as f:
    f.write("# Version 1.0\n")
    f.write(f"# - FULL design: {len(COMMON)} points, NO QA applied (dropping is a config choice)\n")
    f.write("# - Values are LINEAR (not log) for C1/C2/C3\n")
    f.write("# Parameter AlphaS Q0 C1 C2 Tau0 C3\n")
    f.write("# - Parameter AlphaS: Linear [0.1, 0.5]\n# - Parameter Q0: Linear [1, 10]\n")
    f.write("# - Parameter C1: Log [0.006737946999085467, 10]\n")
    f.write("# - Parameter C2: Log [0.006737946999085467, 10]\n")
    f.write("# - Parameter Tau0: Linear [0.0, 1.5]\n")
    f.write("# - Parameter C3: Log [0.049787068367863944, 100]\n")
    # ORIGINAL design ids in the header -> config design_points_to_exclude uses original ids
    f.write("# Design point indices (row index): " + " ".join(str(i) for i in COMMON) + "\n")
    for r in D[COMMON]:
        f.write(" ".join(f"{v:.18e}" for v in r) + "\n")

pred_hdr = "# " + " ".join(f"design_point{i}" for i in COMMON) + "\n"
n_ok = n_cur = n_stat = 0
skipped, zero_bins = [], []

for name, v in items.items():
    lab = label_of(name, v["Attribute"])
    dfile, pfile = f"{IN}/{v['Data']}", f"{IN}/{v['Prediction']}"
    pv, pe = rows(pfile), rows(f"{IN}/{v['PredictionError']}")
    dat, dcols = rows(dfile), header_cols(dfile)
    keep = [i for i in range(len(dat)) if i not in set(v.get("DataExclude", []))]
    # STAT's yaml also trims leading PREDICTION bins (PredictionExclude) where the prediction carries
    # partially-filled edge bins that its data file lacks (SetupAnalysis.py:147-149, axis=1 = bins).
    # Ignoring it skipped 6 observables (ATLAS 2760 ch 0-5, CMS 5020 ch x3, PHENIX pi0 0-10/40-50).
    pexcl = sorted(set(v.get("PredictionExclude", [])))
    if pexcl:
        pv, pe = np.delete(pv, pexcl, axis=0), np.delete(pe, pexcl, axis=0)

    if pv.shape[0] != len(keep):
        skipped.append(f"{lab}: prediction has {pv.shape[0]} bins but data-minus-exclude has "
                       f"{len(keep)} -- binning unrecoverable from raw sources")
        continue
    pos = {dp: j for j, dp in enumerate(dp_labels(pfile))}
    if not all(dp in pos for dp in COMMON):
        skipped.append(f"{lab}: missing common design columns")
        continue
    csel = [pos[dp] for dp in COMMON]
    pv, pe = pv[:, csel], pe[:, csel]
    kd = dat[keep]
    # STAT data files come in two layouts: "xmin xmax y stat,low stat,high ..." (most) and
    # "x y stat,low stat,high ..." (PHENIX pi0 0-10/40-50). Normalise the latter to xmin/xmax by
    # matching the bin centre to the curated table's edges, so every output Data table is edge-based.
    x_mode = bool(dcols) and dcols[0] == "x"
    cf = glob.glob(f"{CUR}/*/Data__{lab}.dat")
    if x_mode:
        if not cf:
            skipped.append(f"{lab}: STAT file is single-x and no curated table to recover bin edges"); continue
        ch0, cd0 = header_cols(cf[0]), rows(cf[0])
        cc = (cd0[:, 0] + cd0[:, 1]) / 2
        idx = [int(np.argmin(np.abs(cc - kd[i, 0]))) for i in range(len(kd))]
        if not np.allclose(cc[idx], kd[:, 0], atol=1e-6):
            skipped.append(f"{lab}: single-x centres do not match curated bin centres"); continue
        # rebuild kd as [xmin xmax y stat,low stat,high sys...] from the STAT x-mode row (shift by 1)
        kd = np.column_stack([cd0[idx, 0], cd0[idx, 1], kd[:, 1:]])
        dcols = ["xmin", "xmax", *dcols[1:]]

    # ---- systematics: prefer the curated named per-source breakdown on these bins,
    #      taking the STATISTICAL column from the SAME source (see module docstring).
    sys_cols = sys_names = stat_cols = None
    if cf:
        ch, cd = header_cols(cf[0]), rows(cf[0])
        if ch and ch[0].lower() == "xmin":
            cbin = {(round(cd[i, 0], 3), round(cd[i, 1], 3)): i for i in range(len(cd))}
            bsel = [cbin.get((round(kd[i, 0], 3), round(kd[i, 1], 3))) for i in range(len(kd))]
            if all(b is not None for b in bsel):
                sys_names, sys_cols = ch[5:], cd[bsel][:, 5:]
                stat_cols = cd[bsel][:, 3:5]
    src = "curated per-source"
    if sys_cols is None:                      # fall back to STAT's own stat+sys columns
        src = "STAT columns"
        sys_names, sys_cols = dcols[5:], kd[:, 5:]
        stat_cols = kd[:, 3:5]
    n_cur, n_stat = (n_cur + 1, n_stat) if src.startswith("curated") else (n_cur, n_stat + 1)

    # Guard: every bin must carry SOME uncertainty, else the covariance is singular there.
    tot = np.sqrt(stat_cols[:, 0] ** 2 + ((sys_cols ** 2).sum(axis=1) if sys_cols.size else 0.0))
    if np.any(tot <= 0):
        zero_bins.append(f"{lab}: {(tot <= 0).sum()}/{len(tot)} bins have ZERO total uncertainty")

    with open(f"{OUT}/Data/Data__{lab}.dat", "w") as f:
        f.write("# Version 1.0\n")
        f.write(f"# FULL build: bins = data minus DataExclude ({len(dat)}->{len(keep)}); "
                f"systematics = {src}; NO pT cut applied beyond DataExclude\n")
        f.write("# Label xmin xmax y stat,low stat,high " + " ".join(sys_names) + "\n")
        for i in range(len(kd)):
            vals = [kd[i, 0], kd[i, 1], kd[i, 2], stat_cols[i, 0], stat_cols[i, 1]] + list(sys_cols[i])
            f.write(" ".join(f"{x:.18e}" for x in vals) + "\n")

    for tag, arr in (("values", pv), ("errors", pe)):
        with open(f"{OUT}/Prediction/Prediction__exponential__{lab}__{tag}.dat", "w") as f:
            f.write(f"# Version 2.0\n# Data Data_exponential__{lab}.dat\n# Design Design_exponential.dat\n")
            f.write(pred_hdr)
            for i in range(arr.shape[0]):
                f.write(" ".join(f"{x:.18e}" for x in arr[i]) + "\n")
    n_ok += 1

print(f"\nbuilt {n_ok}/{len(items)} observables -> {OUT}")
print(f"  design points per observable: {len(COMMON)}")
print(f"  systematics: curated per-source {n_cur}, STAT-column fallback {n_stat}")
if zero_bins:
    print(f"\n  !! {len(zero_bins)} observables with ZERO-uncertainty bins:")
    for z in zero_bins:
        print("   !", z)
if skipped:
    print(f"\n  SKIPPED {len(skipped)} observables (documented, not silent):")
    for s in skipped:
        print("   -", s)
