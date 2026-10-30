"""West Rapti hydromet data preparation for the WA+ labs."""
from __future__ import annotations
import argparse, glob, os, re
import pandas as pd
OUTLET_ID = 375
PRECIP_SHEET = "24h accumulated Precipitation "
def _parse_dfl(path):
    sid = int(re.search(r"DFL_(\d+)", os.path.basename(path)).group(1))
    dates, vals = [], []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if "," not in line: continue
            d, _, v = line.partition(",")
            try: dt = pd.to_datetime(d.strip(), format="%d/%b/%Y")
            except ValueError: continue
            v = v.strip(); dates.append(dt)
            vals.append(float(v) if v not in ("", "-", "NA") else float("nan"))
    s = pd.Series(vals, index=pd.DatetimeIndex(dates), name=sid).sort_index()
    return s[~s.index.duplicated()]
def discharge_monthly(src, start=None, end=None):
    files = sorted(glob.glob(os.path.join(src, "**", "DFL_*.txt"), recursive=True))
    if not files: raise FileNotFoundError(f"no DFL_*.txt under {src}")
    daily = pd.concat([_parse_dfl(f) for f in files], axis=1)
    if start: daily = daily[daily.index >= pd.Timestamp(start)]
    if end:   daily = daily[daily.index <  pd.Timestamp(end)]
    flow = daily.resample("MS").mean()
    days = flow.index.to_series().dt.daysinmonth.values[:, None]
    vol = flow * days * 86400 / 1e6
    return flow, vol
def _clean_name(col): return re.sub(r"\s*\(mm\)\(Sum\)\s*$", "", str(col)).strip()
def dhm_frames(src, start=None, end=None, min_coverage=0.4, fname="rapti_rain.xlsx"):
    xls = os.path.join(src, "Temp_PPT", fname)
    meta = pd.read_excel(xls, sheet_name="Stations")
    name2id = {str(n).strip(): int(i) for n, i in zip(meta["Name"], meta["Station Index"])}
    rain = pd.read_excel(xls, sheet_name=PRECIP_SHEET)
    tcol = rain.columns[0]; rain[tcol] = pd.to_datetime(rain[tcol], errors="coerce")
    rain = rain.dropna(subset=[tcol]).set_index(tcol)
    ren, keep = {}, []
    for c in rain.columns:
        nm = _clean_name(c)
        if nm in name2id: ren[c] = name2id[nm]; keep.append(c)
    monthly = rain[keep].rename(columns=ren).apply(pd.to_numeric, errors="coerce").resample("MS").sum(min_count=15)
    if start: monthly = monthly[monthly.index >= pd.Timestamp(start)]
    if end:   monthly = monthly[monthly.index <  pd.Timestamp(end)]
    cov = monthly.notna().mean(); good = cov[cov >= min_coverage].index.tolist()
    monthly = monthly[good]
    stations = (meta[meta["Station Index"].isin(good)]
                .rename(columns={"Station Index":"id","Longitude":"lon","Latitude":"lat","Elevation":"elev"})
                [["id","lon","lat","elev"]].reset_index(drop=True))
    monthly.index.name = "date"
    return stations, monthly
def dhm_daily(src, fname="rapti_rain.xlsx"):
    """DAILY gauge precip (mm): index=date, columns=station id — for daily quantile mapping."""
    xls = os.path.join(src, "Temp_PPT", fname)
    meta = pd.read_excel(xls, sheet_name="Stations")
    name2id = {str(n).strip(): int(i) for n, i in zip(meta["Name"], meta["Station Index"])}
    rain = pd.read_excel(xls, sheet_name=PRECIP_SHEET)
    tcol = rain.columns[0]; rain[tcol] = pd.to_datetime(rain[tcol], errors="coerce")
    rain = rain.dropna(subset=[tcol]).set_index(tcol)
    ren = {c: name2id[_clean_name(c)] for c in rain.columns if _clean_name(c) in name2id}
    daily = rain[list(ren)].rename(columns=ren).apply(pd.to_numeric, errors="coerce")
    daily.index.name = "date"
    return daily


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", default=".")
    ap.add_argument("--start", default=None); ap.add_argument("--end", default=None)
    ap.add_argument("--outlet", type=int, default=OUTLET_ID)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    stations, precip = dhm_frames(a.src, a.start, a.end)
    stations.to_csv(os.path.join(a.out,"dhm_stations.csv"), index=False)
    precip.to_csv(os.path.join(a.out,"dhm_precip.csv"))
    print(f"dhm_stations.csv: {len(stations)} gauges | dhm_precip.csv: {precip.shape[0]}x{precip.shape[1]}")
    dsrc = os.path.join(a.src,"DailyDischargeRapti")
    flow, vol = discharge_monthly(dsrc if os.path.isdir(dsrc) else a.src, a.start, a.end)
    flow.add_suffix("_m3s").join(vol.add_suffix("_MCM")).to_csv(os.path.join(a.out,"discharge_monthly.csv"))
    out = a.outlet if a.outlet in flow.columns else flow.columns[-1]
    pd.DataFrame({"flow_m3s":flow[out],"vol_MCM":vol[out]}).to_csv(os.path.join(a.out,"discharge_outlet.csv"))
    print(f"discharge stations {list(flow.columns)} | outlet={out}")
if __name__ == "__main__": main()
