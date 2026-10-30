"""
west_rapti — tested, importable helpers for the WA+ labs (basin set in the notebooks).
Keep reusable logic here; notebooks import it so every team runs the SAME code.
Verified with pysheds 0.5 on numpy 2 (via the np.in1d shim below), rioxarray,
pyproj. Runs on Colab's default Python 3.13 + NumPy 2 — do NOT downgrade NumPy.
No GDAL CLI needed.
"""
import json, os, shutil
import pandas as pd
import numpy as np
import rioxarray
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from pyproj import Transformer

# NumPy 2 removed np.in1d, but pysheds 0.5 still calls it during flow routing.
# Restore it as an alias so Lab 1 delineation runs on Colab's native NumPy 2 —
# this is why the labs no longer pin numpy<2 (that build breaks on Python 3.13).
if not hasattr(np, "in1d"):
    np.in1d = np.isin

# NOTE: pysheds is imported lazily inside delineate()/suggest_outlet() so that
# Labs 2-6 can `import west_rapti` without pysheds installed (only Lab 1 needs it).


def reproject_dem(dem_in="glo30.tif", utm="EPSG:32644", res=30, out="dem_utm.tif"):
    """Reproject any-CRS DEM to UTM at `res` metres. Returns the output path."""
    da = rioxarray.open_rasterio(dem_in, masked=True).squeeze("band", drop=True)
    da = da.rio.reproject(utm, resolution=res, resampling=Resampling.bilinear)
    da.rio.to_raster(out)
    return out


def gauge_to_utm(lon, lat, utm="EPSG:32644"):
    """Outlet gauge lon/lat (degrees) -> projected (x, y) metres."""
    return Transformer.from_crs("EPSG:4326", utm, always_xy=True).transform(lon, lat)


# Hydrological conditioning + flow routing is the slow part of delineation. It is cached
# per DEM file (keyed on path + modification time), so the interactive Delineate button,
# suggest_outlet() and check_conditioning() only compute it once per DEM.
_HYDRO_CACHE = {}

def _conditioned(dem_utm, fill=True):
    """Fill pits -> fill depressions -> resolve flats, then D8 flow direction + accumulation.
    Cached per (file, mtime, fill). fill=False skips conditioning (teaching demo only)."""
    from pysheds.grid import Grid   # lazy: only Lab 1 needs pysheds
    key = (os.path.abspath(dem_utm), os.path.getmtime(dem_utm), bool(fill))
    if key not in _HYDRO_CACHE:
        grid = Grid.from_raster(dem_utm)
        dem = grid.read_raster(dem_utm)
        if fill:
            flooded = grid.fill_depressions(grid.fill_pits(dem))
            inflated = grid.resolve_flats(flooded)
        else:
            flooded = inflated = dem
        fdir = grid.flowdir(inflated)
        acc = grid.accumulation(fdir)
        _HYDRO_CACHE[key] = dict(grid=grid, dem=dem, flooded=flooded, inflated=inflated,
                                 fdir=fdir, acc=acc, extent=grid.extent)
    return _HYDRO_CACHE[key]


def delineate(dem_utm="dem_utm.tif", gauge_lon=None, gauge_lat=None,
              utm="EPSG:32644", acc_min=3000, out="basin_mask_30m.tif", fill=True):
    """
    Delineate the basin draining to the outlet gauge, on the hydrologically conditioned
    (filled) DEM. Returns a dict with the intermediate arrays (for plotting), the mask path,
    and `mask` — the basin as a full-DEM-extent 0/1 array (the file at `out` is clipped to
    the basin, as before). fill=False delineates on the raw DEM (teaching demo only).
    """
    c = _conditioned(dem_utm, fill)
    grid, fdir, acc = c["grid"], c["fdir"], c["acc"]
    gx, gy = gauge_to_utm(gauge_lon, gauge_lat, utm)
    streams = acc > acc_min
    if not np.asarray(streams).any():
        raise ValueError(f"No cell has more than acc_min={acc_min} upstream cells"
                         + ("" if fill else " (raw DEM: sinks break the stream network)")
                         + " — lower acc_min, or check the DEM.")
    xs, ys = grid.snap_to_mask(streams, (gx, gy))
    catch = grid.catchment(x=xs, y=ys, fdir=fdir, xytype="coordinate")
    mask = np.asarray(catch).astype("uint8")
    if int(mask.sum()) == 0:
        raise RuntimeError("No basin found — lower acc_min, or check the gauge is inside the DEM.")
    grid.clip_to(catch)
    grid.to_raster(grid.view(catch, dtype="uint8"), out)
    grid.viewfinder = c["dem"].viewfinder     # un-clip the cached grid for the next call
    return dict(dem=np.asarray(c["dem"]), fdir=np.asarray(fdir), acc=np.asarray(acc),
                extent=c["extent"], gauge=(gx, gy), snap=(xs, ys), mask_path=out, mask=mask)


def check_conditioning(dem_utm="dem_utm.tif", filled_out=None, band=1):
    """
    Verify the DEM that delineate() routes on is hydrologically clean.
    Re-uses delineate()'s own conditioning (cached), then checks it independently with a
    morphological-reconstruction fill (scikit-image): re-filling a clean DEM changes nothing.
    Returns dict(cells, filled_cells, filled_pct, max_fill_m, sinks_before, sinks_after,
    no_flowdir_cells, depth). no_flowdir_cells ignores cells within `band` of nodata or the
    DEM edge (those legitimately drain out of the domain). Optionally writes the filled DEM.
    """
    c = _conditioned(dem_utm, True)
    z = np.asarray(c["dem"], dtype="float64")
    nod = c["dem"].nodata
    nmask = ~np.isfinite(z) | ((z == nod) if nod is not None else False)
    flooded = np.asarray(c["flooded"], dtype="float64")
    inflated = np.asarray(c["inflated"], dtype="float64")
    depth = np.where(nmask, 0.0, flooded - z)

    def _residual(a):
        try:
            from skimage.morphology import reconstruction
        except ImportError:
            return None
        work = np.where(nmask, np.nanmin(a[~nmask]) - 1.0, a)
        seed = np.where(nmask, work, work.max())
        seed[0, :], seed[-1, :], seed[:, 0], seed[:, -1] = work[0, :], work[-1, :], work[:, 0], work[:, -1]
        r = reconstruction(seed, work, method="erosion") - work
        return int(((r > 1e-3) & ~nmask).sum())

    near = nmask.copy()                          # dilate nodata by `band` cells (8-neighbour)
    for _ in range(band):
        pad = np.pad(near, 1, constant_values=False)
        near = near | np.logical_or.reduce([pad[1 + dy:pad.shape[0] - 1 + dy, 1 + dx:pad.shape[1] - 1 + dx]
                                            for dy in (-1, 0, 1) for dx in (-1, 0, 1)])
    near[:band, :] = near[-band:, :] = True
    near[:, :band] = near[:, -band:] = True
    fdir = np.asarray(c["fdir"])
    stats = dict(cells=int((~nmask).sum()),
                 filled_cells=int((depth > 1e-6).sum()),
                 filled_pct=round(float(100 * (depth > 1e-6).sum() / max(1, (~nmask).sum())), 3),
                 max_fill_m=round(float(depth.max()), 2),
                 sinks_before=_residual(z), sinks_after=_residual(inflated),
                 no_flowdir_cells=int(((fdir <= 0) & ~nmask & ~near).sum()),
                 depth=depth)
    if filled_out:
        c["grid"].to_raster(c["flooded"], filled_out)
    return stats


def build_wa_grid(dem_utm="dem_utm.tif", wa_res=300, utm="EPSG:32644", out="gridspec.json", te=None):
    """Define the canonical WA+ grid (snap DEM bounds to wa_res). SHARE with all teams.
    te=[xmin, ymin, xmax, ymax] pins the grid to a fixed extent instead (e.g. to keep an
    existing grid when the DEM covers only the basin) — it must be a multiple of wa_res."""
    if te is None:
        minx, miny, maxx, maxy = rioxarray.open_rasterio(dem_utm).rio.bounds()
    else:
        minx, miny, maxx, maxy = [float(v) for v in te]
    xmin = np.floor(minx / wa_res) * wa_res; ymin = np.floor(miny / wa_res) * wa_res
    xmax = np.ceil(maxx / wa_res) * wa_res;  ymax = np.ceil(maxy / wa_res) * wa_res
    gs = dict(crs=utm, res=wa_res, te=[float(xmin), float(ymin), float(xmax), float(ymax)],
              width=int((xmax - xmin) / wa_res), height=int((ymax - ymin) / wa_res))
    json.dump(gs, open(out, "w"), indent=2)
    return gs


def warp_mask(mask30="basin_mask_30m.tif", gridspec=None, utm="EPSG:32644",
              tif="basin_mask_300m.tif", nc="basin_mask.nc"):
    """Warp the 30 m mask onto the canonical grid; write GeoTIFF + NetCDF. Returns the DataArray."""
    tr = from_origin(gridspec["te"][0], gridspec["te"][3], gridspec["res"], gridspec["res"])
    m = rioxarray.open_rasterio(mask30).squeeze("band", drop=True)
    m300 = m.rio.reproject(utm, transform=tr, shape=(gridspec["height"], gridspec["width"]),
                           resampling=Resampling.nearest)
    m300.rio.to_raster(tif)
    m300.name = "basin_mask"
    m300.to_netcdf(nc)
    return m300


def write_basin(out="basin_config.json", name="West_Rapti", hydroyear=10):
    """Write the BASIN skeleton (placeholder data-cube paths, filled across the week)."""
    BASIN = {
        "name": name, "hydroyear": hydroyear, "chunksize": 1000, "unit_conversion": 1e3,
        "gis_data": {"basin_mask": "basin_mask.nc", "dem": "dem_utm.tif",
                     "aeisw": None, "population": None, "wpl": None, "environ_water_req": None},
        "data_cube": {"monthly": {k: None for k in
            ["p","et","etref","i","nrd","lai","ndm","sro","etincr","etrain","perc"]}}}
    json.dump(BASIN, open(out, "w"), indent=2)
    return BASIN


def suggest_outlet(dem_utm="dem_utm.tif", utm="EPSG:32644"):
    """
    Suggest an outlet: the point where the largest river LEAVES the DEM.
    Returns {lon, lat, x, y, acc}. Use it if you don't have a gauge coordinate,
    or to sanity-check one. Paste lon/lat into GAUGE_LON, GAUGE_LAT.
    """
    c = _conditioned(dem_utm, True)
    grid, acc = c["grid"], np.asarray(c["acc"])
    edge = np.zeros(acc.shape, bool)
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True   # domain boundary
    r, c = np.unravel_index(np.argmax(np.where(edge, acc, -1)), acc.shape)
    x, y = grid.affine * (c + 0.5, r + 0.5)
    lon, lat = Transformer.from_crs(utm, "EPSG:4326", always_xy=True).transform(x, y)
    return {"lon": round(float(lon), 5), "lat": round(float(lat), 5),
            "x": float(x), "y": float(y), "acc": float(acc[r, c])}


def save_outputs(files, out_dir):
    """Copy the given output files to out_dir (e.g. a Drive folder) so they
    survive a Colab disconnect and feed the next lab. Returns what was saved."""
    os.makedirs(out_dir, exist_ok=True)
    saved = []
    for f in files:
        if os.path.exists(f):
            shutil.copy(f, out_dir); saved.append(os.path.basename(f))
    return saved


def load_inputs(files, in_dir):
    """Copy files from in_dir into the working dir if not already present.
    Use at the top of a later lab to pull the previous lab's outputs. Returns what was pulled."""
    got = []
    for f in files:
        src = os.path.join(in_dir, f)
        if os.path.exists(src) and not os.path.exists(f):
            shutil.copy(src, f); got.append(f)
    return got


# ============================ Lab 2 helpers ============================
def lonlat_bbox(gridspec, pad=0.02):
    """Return [W, S, E, N] in degrees for the canonical grid (for GEE filtering/clip)."""
    xmin, ymin, xmax, ymax = gridspec["te"]
    tr = Transformer.from_crs(gridspec["crs"], "EPSG:4326", always_xy=True)
    lons, lats = [], []
    for x in (xmin, xmax):
        for y in (ymin, ymax):
            lo, la = tr.transform(x, y); lons.append(lo); lats.append(la)
    return [min(lons)-pad, min(lats)-pad, max(lons)+pad, max(lats)+pad]


def regrid(src, gridspec, resampling="bilinear"):
    """Reproject a raster (path or DataArray, any #bands) onto the canonical WA+ grid.

    Reads files with masked=True so a GeoTIFF's declared nodata becomes NaN, and
    scrubs any non-finite sentinel (geemap/EE writes masked pixels as +/-inf or a
    float-min fill for MODIS/WaPOR exports). Without this, those sentinels survive
    the reprojection and poison basin means (the classic 'ET = -inf' at closure).
    Integer rasters (e.g. land-cover class maps) keep their integer dtype, but an
    out-of-range inherited nodata is normalised so the reproject cannot overflow.
    """
    da = rioxarray.open_rasterio(src, masked=True) if isinstance(src, str) else src
    if np.issubdtype(da.dtype, np.floating):
        da = da.where(np.isfinite(da))
        nodata = np.nan
    else:
        # Integer raster (e.g. a class map). A downloaded map often carries an int32-min
        # nodata (-2147483648); when the array has been recast to int16 (WA+ land use), even
        # READING .rio.nodata raises OverflowError, and reproject would too. So strip any
        # inherited nodata/_FillValue without touching the throwing getter, then set a safe
        # in-range fill (0 = background for the WA+ land-use codes 1..9).
        da = da.copy()
        for holder in (da.attrs, da.encoding):
            holder.pop("_FillValue", None); holder.pop("nodata", None)
        try:
            da = da.rio.write_nodata(None)
        except Exception:
            pass
        nodata = 0
        da = da.rio.write_nodata(nodata)
    tr = from_origin(gridspec["te"][0], gridspec["te"][3], gridspec["res"], gridspec["res"])
    out = da.rio.reproject(gridspec["crs"], transform=tr,
                           shape=(gridspec["height"], gridspec["width"]),
                           resampling=getattr(Resampling, resampling),
                           nodata=nodata)
    if np.issubdtype(out.dtype, np.floating):
        out = out.where(np.isfinite(out))
    return out


def tif_stack_to_nc(tif, name, dates, gridspec, nc=None, resampling="bilinear"):
    """Multiband GeoTIFF (band = month) -> monthly DataArray on the WA+ grid with a time axis.

    Tolerates an off-by-one between the raster's band count and `dates`: some products
    return one extra month (e.g. WaPOR AETI includes the END month while DATES excludes
    it), so both the bands and the dates are trimmed to the shorter length instead of
    raising a coordinate-size mismatch.
    """
    import pandas as pd
    dates = pd.to_datetime(list(dates))
    da = regrid(tif, gridspec, resampling)
    n = min(int(da.sizes["band"]), len(dates))
    da = da.isel(band=slice(0, n)).rename({"band": "time"}).assign_coords(time=dates[:n])
    da.name = name
    da.rio.write_crs(gridspec["crs"], inplace=True)
    if nc:
        da.to_netcdf(nc)
    return da


def glacier_mask_from_vector(vector_path, gridspec, out_nc=None, where=None):
    """
    Rasterise a real glacier-outline vector (RGI / GLIMS / ICIMOD shapefile or GeoJSON)
    onto the WA+ grid -> a 0/1 glacier mask DataArray. Polygons are reprojected to the
    grid CRS first. `where`: optional (column, min_value) to drop tiny glaciers,
    e.g. ("area_km2", 0.05). Feed the result to snow_module(..., glacier=mask).
    """
    import geopandas as gpd
    from rasterio.features import rasterize
    gdf = gpd.read_file(vector_path)
    if where is not None:
        col, vmin = where
        if col in gdf.columns:
            gdf = gdf[gdf[col] >= vmin]
    gdf = gdf.to_crs(gridspec["crs"])
    xmin, ymin, xmax, ymax = gridspec["te"]; res = gridspec["res"]
    w, h = int(gridspec["width"]), int(gridspec["height"])
    transform = from_origin(xmin, ymax, res, res)
    arr = rasterize(((geom, 1) for geom in gdf.geometry if geom is not None),
                    out_shape=(h, w), transform=transform, fill=0, dtype="int16", all_touched=True)
    xs = xmin + (np.arange(w) + 0.5) * res
    ys = ymax - (np.arange(h) + 0.5) * res
    da = xr.DataArray(arr, dims=("y", "x"), coords={"y": ys, "x": xs}, name="glacier")
    da.rio.write_crs(gridspec["crs"], inplace=True)
    if out_nc:
        da.to_netcdf(out_nc)
    return da


def dekads_to_monthly(src_tif, out_tif, n_months, how="sum"):
    """
    Aggregate a chronological multi-band dekadal GeoTIFF (e.g. WaPOR L1-NPP-D over
    a whole-month range: 3 dekads per month) into an n_months-band GeoTIFF.
    Bands are split into n_months contiguous groups and summed (or averaged).
    Used by the NDM_SOURCE='wapor' path in Lab 2.
    """
    da = rioxarray.open_rasterio(src_tif, masked=True)          # (band, y, x)
    nb = int(da.sizes["band"])
    edges = np.linspace(0, nb, n_months + 1).astype(int)
    parts = []
    for k in range(n_months):
        sl = da.isel(band=slice(edges[k], edges[k + 1]))
        parts.append(sl.sum("band") if how == "sum" else sl.mean("band"))
    stack = xr.concat(parts, dim="band").assign_coords(band=np.arange(1, n_months + 1))
    try:
        stack.rio.write_crs(da.rio.crs, inplace=True)
    except Exception:
        pass
    stack.rio.to_raster(out_tif)
    return out_tif


def basin_series(da, mask_da, how="mean"):
    """Reduce a (time,y,x) cube to a monthly basin series over mask==1. Returns a pandas Series."""
    ydim, xdim = da.dims[-2], da.dims[-1]
    d = da.where(mask_da == 1).where(np.isfinite(da))   # drop any non-finite sentinel first
    s = d.mean(dim=[ydim, xdim]) if how == "mean" else d.sum(dim=[ydim, xdim])
    return s.to_pandas()


def _station_pixel_values(da, stations, utm):
    import pandas as pd
    tr = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    xdim, ydim = da.dims[-1], da.dims[-2]
    out = {}
    for _, s in stations.iterrows():
        x, y = tr.transform(s["lon"], s["lat"])
        out[s["id"]] = da.sel({xdim: x, ydim: y}, method="nearest").values
    return pd.DataFrame(out, index=pd.to_datetime(da["time"].values))


def _gauge_xy(stations, utm):
    from pyproj import Transformer
    tr = Transformer.from_crs("EPSG:4326", utm, always_xy=True)
    x, y = tr.transform(np.asarray(stations["lon"], float), np.asarray(stations["lat"], float))
    return np.asarray(x, float), np.asarray(y, float)


def _idw(gx, gy, gf, X, Y, power=2.0):
    """Inverse-distance interpolation of gauge factors gf at (gx,gy) onto grid (X,Y)."""
    num = np.zeros_like(X, float); den = np.zeros_like(X, float)
    for xi, yi, fi in zip(gx, gy, gf):
        if not np.isfinite(fi):
            continue
        w = 1.0 / np.power((X - xi) ** 2 + (Y - yi) ** 2 + 1.0, power / 2.0)
        num += w * fi; den += w
    return np.divide(num, den, out=np.ones_like(X), where=den > 0)


def _apply_factor_field(P_da, stations, fac, utm, fmin, fmax, idw_power):
    """IDW the per-gauge per-month factor array (ngauge x 12) onto the grid and apply by month."""
    yd, xd = P_da.dims[-2], P_da.dims[-1]
    gx, gy = _gauge_xy(stations, utm)
    X, Y = np.meshgrid(P_da[xd].values, P_da[yd].values)
    Fm = {m: np.clip(_idw(gx, gy, fac[:, m - 1], X, Y, idw_power), fmin, fmax) for m in range(1, 13)}
    out = P_da.copy()
    mon = pd.DatetimeIndex(P_da["time"].values).month
    for ti in range(P_da.sizes["time"]):
        out[ti] = P_da[ti] * Fm[int(mon[ti])]
    return out.rename("P")


def _monthly_factors(sat, obs, ids, min_mm=10.0, fmin=0.25, fmax=4.0):
    """Per-gauge, per-calendar-month factor = sum(obs)/sum(sat) over the record (monthly totals)."""
    fac = np.ones((len(ids), 12))
    for gi, sid in enumerate(ids):
        if sid not in obs.columns:
            continue
        o = obs[sid].reindex(sat.index).values; s = sat[sid].values
        base = np.isfinite(o) & np.isfinite(s) & (o >= min_mm) & (s >= min_mm)
        for m in range(1, 13):
            sel = base & (sat.index.month == m)
            if sel.any() and s[sel].sum() > 0:
                fac[gi, m - 1] = np.clip(o[sel].sum() / s[sel].sum(), fmin, fmax)
            elif base.any() and s[base].sum() > 0:
                fac[gi, m - 1] = np.clip(o[base].sum() / s[base].sum(), fmin, fmax)
    return fac


def _qm_apply(x, sref, gref):
    """Empirical quantile mapping: map values x (satellite) onto the gauge distribution."""
    ss = np.sort(np.asarray(sref, float)[np.isfinite(sref)])
    gg = np.sort(np.asarray(gref, float)[np.isfinite(gref)])
    if len(ss) < 2 or len(gg) < 2:
        return np.asarray(x, float)
    q = np.interp(np.asarray(x, float), ss, np.linspace(0, 1, len(ss)))
    return np.interp(q, np.linspace(0, 1, len(gg)), gg)


def quantile_factors(stations, gauge_daily, sat_daily, min_days=30, fmin=0.25, fmax=4.0):
    """DAILY quantile mapping per gauge per calendar month -> a monthly multiplicative factor
    (sum of QM-corrected daily / sum of raw daily). gauge_daily, sat_daily: daily DataFrames,
    index=date, columns=station id, over the overlap period."""
    stations, gauge_daily, sat_daily = _harmonise_ids(stations, gauge_daily, sat_daily)
    ids = list(stations["id"]); fac = np.ones((len(ids), 12))
    gcols = set(getattr(gauge_daily, "columns", [])); scols = set(getattr(sat_daily, "columns", []))
    for gi, sid in enumerate(ids):
        if sid not in gcols or sid not in scols:
            continue
        common = gauge_daily.index.intersection(sat_daily.index)
        g = gauge_daily[sid].reindex(common).values
        s = sat_daily[sid].reindex(common).values
        mon = pd.DatetimeIndex(common).month
        for m in range(1, 13):
            sel = (mon == m) & np.isfinite(g) & np.isfinite(s)
            if sel.sum() >= min_days and s[sel].sum() > 0:
                corr = _qm_apply(s[sel], s[sel], g[sel])
                fac[gi, m - 1] = np.clip(corr.sum() / s[sel].sum(), fmin, fmax)
    return fac


def _elev_idw_field(stations, fac_m, elev_da, utm, fmin, fmax, idw_power):
    """Per-month factor field = elevation trend of the gauge factors + IDW of the residuals.
    Captures the altitude-driven CHIRPS bias AND the local spatial pattern."""
    gx, gy = _gauge_xy(stations, utm)
    gz = np.asarray(stations["elev"], float)
    yd, xd = elev_da.dims[-2], elev_da.dims[-1]
    X, Y = np.meshgrid(elev_da[xd].values, elev_da[yd].values)
    Z = np.asarray(elev_da.values, float)
    out = {}
    for m in range(1, 13):
        f = fac_m[:, m - 1]
        ok = np.isfinite(f) & np.isfinite(gz)
        if ok.sum() >= 3 and np.ptp(gz[ok]) > 0:                 # elevation trend of the factor
            b, a = np.polyfit(gz[ok], f[ok], 1)
            trend_grid = a + b * Z
            resid = f - (a + b * gz)                              # what elevation doesn't explain
        else:
            trend_grid = np.full(Z.shape, float(np.nanmean(f[ok])) if ok.any() else 1.0)
            resid = f - np.nanmean(f[ok]) if ok.any() else np.zeros_like(f)
        resid_grid = _idw(gx[ok], gy[ok], resid[ok], X, Y, idw_power)   # IDW the residual
        out[m] = np.clip(trend_grid + resid_grid, fmin, fmax)
    return out

def _harmonise_ids(stations, *frames):
    """Gauge ids as TEXT everywhere. dhm_precip.csv stores ids as column NAMES (read back as text) while
    dhm_stations.csv stores them as numbers; without this no gauge matches and every factor stays 1."""
    st = stations.copy(); st["id"] = st["id"].astype(str).str.replace(r"\.0$", "", regex=True)
    out = []
    for f in frames:
        if f is None: out.append(None); continue
        f = f.copy(); f.columns = [str(c).replace(".0", "") if str(c).endswith(".0") else str(c) for c in f.columns]
        out.append(f)
    return (st, *out)


def precip_bias_table(P_da, stations, obs, utm="EPSG:32644", min_mm=10.0):
    """Long table of gauge vs satellite monthly totals per gauge and calendar month:
    gauge, elev, month, obs_mm, sat_mm, ratio (= sum obs / sum sat over the record), n_months."""
    stations, obs = _harmonise_ids(stations, obs)
    sat = _station_pixel_values(P_da, stations, utm)
    rows = []
    for _, s in stations.iterrows():
        sid = s["id"]
        if sid not in obs.columns: continue
        o = obs[sid].reindex(sat.index).values; v = sat[sid].values
        base = np.isfinite(o) & np.isfinite(v) & (o >= min_mm) & (v >= min_mm)
        for m in range(1, 13):
            sel = base & (sat.index.month == m)
            if sel.any():
                rows.append({"gauge": sid, "elev": float(s.get("elev", np.nan)), "month": m,
                             "obs_mm": float(o[sel].sum()), "sat_mm": float(v[sel].sum()),
                             "ratio": float(o[sel].sum() / v[sel].sum()) if v[sel].sum() > 0 else np.nan,
                             "n_months": int(sel.sum())})
    return pd.DataFrame(rows)


def bias_correct_precip(P_da, stations, obs, dem_da=None, utm="EPSG:32644",
                        method="monthly", min_mm=10.0, fmin=0.25, fmax=4.0, idw_power=2.0,
                        gauge_daily=None, sat_daily=None):
    """
    Multiplicative bias correction of a monthly satellite precip cube against sparse gauges.

    method='monthly'  (default): per-CALENDAR-MONTH factor per gauge = sum(obs)/sum(sat), IDW to grid.
    method='annual'   : one factor per gauge (all months pooled), IDW.
    method='quantile' : DAILY empirical quantile mapping (needs gauge_daily + sat_daily over the
                        overlap) -> monthly factor per gauge, IDW to grid. Corrects the intensity
                        distribution, not just the mean; best when the satellite bias is conditional.
    method='elevation': legacy per-timestep ratio-vs-elevation regression (needs dem_da).
    """
    stations, obs, gauge_daily, sat_daily = _harmonise_ids(stations, obs, gauge_daily, sat_daily)
    sat = _station_pixel_values(P_da, stations, utm)
    ids = list(stations["id"])
    yd, xd = P_da.dims[-2], P_da.dims[-1]

    if method == "quantile":
        if gauge_daily is None or sat_daily is None:
            raise ValueError("method='quantile' needs gauge_daily and sat_daily (daily DataFrames).")
        fac = quantile_factors(stations, gauge_daily, sat_daily, fmin=fmin, fmax=fmax)
        return _apply_factor_field(P_da, stations, fac, utm, fmin, fmax, idw_power)

    if method in ("monthly", "annual"):
        fac = _monthly_factors(sat, obs, ids, min_mm, fmin, fmax)
        if method == "annual":
            fac = fac.mean(axis=1, keepdims=True) * np.ones((1, 12))
        return _apply_factor_field(P_da, stations, fac, utm, fmin, fmax, idw_power)

    if method == "elev_idw":
        if dem_da is None:
            raise ValueError("method='elev_idw' needs dem_da (a DEM on the precip grid).")
        fac = _monthly_factors(sat, obs, ids, min_mm, fmin, fmax)
        _dem = dem_da.squeeze(drop=True)
        ref = P_da.isel(time=0)
        if tuple(_dem.shape[-2:]) != tuple(ref.shape[-2:]):
            try: _dem = _dem.rio.write_crs(utm).rio.reproject_match(ref.rio.write_crs(utm))
            except Exception:
                yd, xd = P_da.dims[-2], P_da.dims[-1]
                _dem = _dem.interp({_dem.dims[-1]: ref[xd].values, _dem.dims[-2]: ref[yd].values})
        Fm = _elev_idw_field(stations, fac, _dem, utm, fmin, fmax, idw_power)
        out = P_da.copy(); mon = pd.DatetimeIndex(P_da["time"].values).month
        for ti in range(P_da.sizes["time"]):
            out[ti] = P_da[ti] * Fm[int(mon[ti])]
        return out.rename("P")

    # ---- legacy elevation-regression method ----
    z = stations.set_index("id")["elev"]
    ref = P_da.isel(time=0)
    try: ref = ref.rio.write_crs(utm)
    except Exception: pass
    dref = dem_da.squeeze(drop=True)
    if tuple(dref.shape[-2:]) != tuple(ref.shape[-2:]):
        try: dref = dref.rio.write_crs(utm).rio.reproject_match(ref)
        except Exception:
            dref = dref.interp({dref.dims[-1]: ref[xd].values, dref.dims[-2]: ref[yd].values})
    Z = np.asarray(dref.values, float)
    corr = P_da.copy()
    for ti, t in enumerate(P_da["time"].values):
        t = pd.Timestamp(t); r = {}
        for sid in ids:
            if sid in obs.columns and t in obs.index:
                o = obs.loc[t, sid]; sv = sat.loc[t, sid]
                if np.isfinite(o) and np.isfinite(sv) and sv > 0: r[sid] = o / sv
        zz = np.array([z.get(k, np.nan) for k in r], float); rr = np.array([r[k] for k in r], float)
        ok = np.isfinite(zz) & np.isfinite(rr); zz, rr = zz[ok], rr[ok]
        if len(rr) >= 3 and np.ptp(zz) > 0:
            try: slope, inter = np.polyfit(zz, rr, 1); field = np.clip(inter + slope * Z, fmin, fmax)
            except (np.linalg.LinAlgError, ValueError): field = np.clip(np.full(Z.shape, float(rr.mean())), fmin, fmax)
        elif len(rr) >= 1: field = np.clip(np.full(Z.shape, float(rr.mean())), fmin, fmax)
        else: field = np.ones(Z.shape)
        corr[ti] = P_da[ti] * field
    return corr.rename("P")


def loo_precip(P_da, stations, obs, utm="EPSG:32644", method="monthly", qfac=None,
               min_mm=10.0, fmin=0.25, fmax=4.0, idw_power=2.0):
    """Leave-one-GAUGE-out validation. For method='quantile', pass qfac (from quantile_factors)."""
    stations, obs = _harmonise_ids(stations, obs)
    sat = _station_pixel_values(P_da, stations, utm)
    ids = list(stations["id"])

    if method == "elev_idw":
        gx, gy = _gauge_xy(stations, utm)
        gz = np.asarray(stations["elev"], float)
        fac = _monthly_factors(sat, obs, ids, min_mm, fmin, fmax)
        errs = []
        for hi, hid in enumerate(ids):
            if hid not in obs.columns: continue
            keep = [j for j in range(len(ids)) if j != hi]
            o = obs[hid].reindex(sat.index).values; s2 = sat[hid].values
            for m in range(1, 13):
                f = fac[keep, m - 1]; zk = gz[keep]
                ok = np.isfinite(f) & np.isfinite(zk)
                if ok.sum() >= 3 and np.ptp(zk[ok]) > 0:
                    b, a = np.polyfit(zk[ok], f[ok], 1)
                    trend = a + b * gz[hi]; resid = f - (a + b * zk)
                else:
                    trend = float(np.nanmean(f[ok])) if ok.any() else 1.0; resid = f * 0.0
                rg = float(_idw(gx[keep][ok], gy[keep][ok], resid[ok],
                                np.array([[gx[hi]]]), np.array([[gy[hi]]]), idw_power)[0, 0])
                F = float(np.clip(trend + rg, fmin, fmax))
                sel = (sat.index.month == m) & np.isfinite(o) & (s2 > 0) & (o >= min_mm)
                for k in np.where(sel)[0]:
                    errs.append(s2[k] * F - o[k])
        errs = np.array(errs)
        return {"rmse": float(np.sqrt(np.mean(errs**2))) if errs.size else float("nan"),
                "bias": float(errs.mean()) if errs.size else float("nan"), "n": int(errs.size)}

    if method in ("monthly", "annual", "quantile"):
        gx, gy = _gauge_xy(stations, utm)
        if method == "quantile":
            fac = qfac if qfac is not None else _monthly_factors(sat, obs, ids, min_mm, fmin, fmax)
        else:
            fac = _monthly_factors(sat, obs, ids, min_mm, fmin, fmax)
            if method == "annual":
                fac = fac.mean(axis=1, keepdims=True) * np.ones((1, 12))
        errs = []
        for hi, hid in enumerate(ids):
            if hid not in obs.columns: continue
            keep = [j for j in range(len(ids)) if j != hi]
            o = obs[hid].reindex(sat.index).values; s = sat[hid].values
            for m in range(1, 13):
                F = float(np.clip(_idw(gx[keep], gy[keep], fac[keep, m - 1],
                          np.array([[gx[hi]]]), np.array([[gy[hi]]]), idw_power)[0, 0], fmin, fmax))
                sel = (sat.index.month == m) & np.isfinite(o) & (s > 0) & (o >= min_mm)
                for k in np.where(sel)[0]:
                    errs.append(s[k] * F - o[k])
        errs = np.array(errs)
        return {"rmse": float(np.sqrt(np.mean(errs**2))) if errs.size else float("nan"),
                "bias": float(errs.mean()) if errs.size else float("nan"), "n": int(errs.size)}

    # legacy elevation LOO
    z = stations.set_index("id")["elev"]; errs = []
    for t in obs.index:
        r = {sid: obs.loc[t, sid] / sat.loc[t, sid] for sid in ids
             if sid in obs.columns and np.isfinite(obs.loc[t, sid]) and sat.loc[t, sid] > 0}
        if len(r) < 3: continue
        for held in list(r):
            if not np.isfinite(z.get(held, np.nan)): continue
            others = [k for k in r if k != held and np.isfinite(z.get(k, np.nan))]
            if len(others) < 2: continue
            zz = np.array([z[k] for k in others], float); rr = np.array([r[k] for k in others], float)
            try: slope, inter = np.polyfit(zz, rr, 1) if np.ptp(zz) > 0 else (0.0, rr.mean())
            except (np.linalg.LinAlgError, ValueError): slope, inter = 0.0, float(rr.mean())
            fac1 = float(np.clip(inter + slope * z[held], fmin, fmax))
            errs.append(sat.loc[t, held] * fac1 - obs.loc[t, held])
    errs = np.array(errs)
    if errs.size == 0: return {"rmse": float("nan"), "bias": float("nan"), "n": 0}
    return {"rmse": float(np.sqrt(np.mean(errs**2))), "bias": float(np.mean(errs)), "n": int(len(errs))}


# ============================ Lab 3 helpers ============================
# WA+ land-use classes (Nepal-oriented starter). code -> (category, name)
WAPLUS_CLASSES = {
    1: ("Protected", "Glacier & snow"),
    2: ("Protected", "Protected forest / wetland"),
    3: ("Utilized",  "Forest"),
    4: ("Utilized",  "Shrub / grassland"),
    5: ("Utilized",  "Bare / rock"),
    6: ("Modified",  "Rainfed cropland"),
    7: ("Modified",  "Settlement / urban"),
    8: ("Managed",   "Irrigated cropland"),
    9: ("Managed",   "Water body / reservoir"),
}

# ESA WorldCover code -> WA+ code (irrigated split needs the command-area overlay; see notebook)
WORLDCOVER_TO_WAPLUS = {
    10: 3,   # tree cover        -> Forest (Utilized)
    20: 4,   # shrubland         -> Shrub/grass
    30: 4,   # grassland         -> Shrub/grass
    40: 6,   # cropland          -> Rainfed (default; overlay reclasses to 8 Irrigated)
    50: 7,   # built-up          -> Settlement (Modified)
    60: 5,   # bare / sparse     -> Bare/rock (Utilized)
    70: 1,   # snow & ice        -> Glacier & snow (Protected)
    80: 9,   # permanent water   -> Water body (Managed)
    90: 2,   # herbaceous wetland -> Protected wetland
    95: 2,   # mangroves         -> Protected wetland
    100: 4,  # moss & lichen     -> Shrub/grass
}


def accuracy_report(y_true, y_pred, labels=None):
    """Confusion matrix + overall / kappa / per-class producer's & user's accuracy."""
    from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    labels = labels if labels is not None else sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pa = np.divide(np.diag(cm), cm.sum(axis=1))   # producer's = recall
        ua = np.divide(np.diag(cm), cm.sum(axis=0))   # user's = precision
    return {"labels": list(labels), "confusion": cm.astype(int),
            "overall": float(accuracy_score(y_true, y_pred)),
            "kappa": float(cohen_kappa_score(y_true, y_pred)),
            "producers": {int(l): round(float(p), 3) for l, p in zip(labels, np.nan_to_num(pa))},
            "users":     {int(l): round(float(u), 3) for l, u in zip(labels, np.nan_to_num(ua))}}


def remap_landuse(lc, mapping):
    """Remap a land-cover map (DataArray or ndarray) to WA+ codes via a {cover:wa} dict."""
    arr = np.asarray(lc)
    out = np.zeros(arr.shape, dtype="int16")
    for k, v in mapping.items():
        out[arr == k] = v
    if hasattr(lc, "dims"):            # xarray DataArray
        r = lc.copy(data=out); r.name = "lu"; return r
    return out


def apply_command_area(lu, command_mask, irrigated_code=8, crop_code=6):
    """Where the irrigation command-area mask is 1 AND lu is rainfed crop, set Irrigated (Managed)."""
    arr = np.asarray(lu).copy(); cm = np.asarray(command_mask)
    arr[(arr == crop_code) & (cm == 1)] = irrigated_code
    if hasattr(lu, "dims"):            # xarray DataArray
        r = lu.copy(data=arr.astype("int16")); r.name = "lu"; return r
    return arr


def check_same_grid(nc_files, gridspec):
    """Return [(file,(h,w))] for any NetCDF not on the canonical grid; empty list = all aligned."""
    import xarray as xr
    bad = []
    for f in nc_files:
        ds = xr.open_dataset(f); da = ds[list(ds.data_vars)[0]]
        h = da.sizes.get("y") or da.sizes.get("lat")
        w = da.sizes.get("x") or da.sizes.get("lon")
        if (h, w) != (gridspec["height"], gridspec["width"]):
            bad.append((f, (h, w)))
    return bad


def assemble_cube(cube_paths, lu_nc=None, basin_config="basin_config.json"):
    """Write the data-cube + land-use paths into basin_config.json (freezes the ARD wiring)."""
    bc = json.load(open(basin_config))
    for k, v in cube_paths.items():
        bc["data_cube"]["monthly"][k] = v
    if lu_nc:
        bc["gis_data"]["lu"] = lu_nc
    json.dump(bc, open(basin_config, "w"), indent=2)
    return bc


# ============================ Lab 4 helpers (hydroloop) ============================
# Root depth per WA+ land-use class (metres) — edit for Nepal soils.
ROOT_DEPTH_M = {1: 0.05, 2: 1.0, 3: 1.2, 4: 0.5, 5: 0.10, 6: 0.6, 7: 0.2, 8: 0.8, 9: 0.05}


def interception_from_lai(LAI, P, NRD=None, k=0.2):
    """Monthly canopy interception (mm) from LAI.

    Per rain day (von Hoyningen-Huene / Braden form, as in WA+ hydroloop):
        cap = k·LAI  (mm per rain day) ;  P_day = P / NRD
        I_day = cap·(1 − 1/(1 + P_day/cap)) ;  I = I_day·NRD   (bounded 0..P)
    Pass NRD (rain days per month). Without NRD the per-day capacity is applied to the monthly
    total (legacy behaviour) — that badly under-estimates interception; avoid it.
    """
    cap = k * LAI
    capw = cap.where(cap > 0, 1e-6)
    if NRD is not None:
        nrd = NRD.where(NRD >= 1, 1.0)
        pday = P / nrd
        I = cap * (1 - 1 / (1 + pday / capw)) * nrd
    else:
        I = cap * (1 - 1 / (1 + P / capw))
    I = I.where(cap > 0, 0.0)
    return I.where(I <= P, P).clip(min=0)


def smmax_from_lu(lu, root_depth=None, awc=0.12):
    """Max root-zone soil moisture (mm) per pixel = root_depth(m)·1000·AWC.

    root_depth: {WA+ class: metres} (defaults to ROOT_DEPTH_M) — spatial via land use.
    awc: available water content (m3/m3). Either a scalar (uniform, the classic 0.12) OR a
         per-pixel array/DataArray on the same grid as `lu` (e.g. field-capacity minus
         wilting-point from SoilGrids/OpenLandMap) — SMmax then varies with BOTH soil and land use.
    """
    rd = root_depth or ROOT_DEPTH_M
    arr = np.asarray(lu)
    rootm = np.zeros(arr.shape, "float32")               # rooting depth (m) per pixel, by class
    for k, v in rd.items():
        rootm[arr == k] = v
    awc_arr = np.asarray(awc, "float32")                 # scalar broadcasts; array is used per-pixel
    out = (rootm * 1000.0 * awc_arr).astype("float32")
    if hasattr(lu, "dims"):
        return lu.copy(data=out).astype("float32").rename("smmax")
    return out


def run_sm_balance(P, ET, I, NRD, SMmax, cf=1.0, sm_init_frac=0.5, et_includes_interception=True):
    """
    Pixel monthly soil-moisture balance (WA_jordan model_SMBalance logic).

    et_includes_interception=True (default; MOD16 and WaPOR AETI both include wet-canopy evaporation):
      Ei = min(I, ET)  — interception that evaporates is part of the satellite ET and is GREEN water
      Pe = P − Ei ; ET_soil = ET − Ei   (the soil only supplies what the canopy did not)
    et_includes_interception=False: legacy (Pe = P − I, soil supplies all of ET — counts I twice).

      SRO = SCS-CN(Pe, NRD, SMmax, SM, cf) ; infil = Pe − SRO ; green = SM + infil
      ET_green_soil = min(ET_soil, green) ; ET_blue (etincr) = ET_soil − ET_green_soil
      etrain = Ei + ET_green_soil  (so etrain + etincr = ET exactly)
      PERC = max(green − ET_green_soil − SMmax, 0) ; SM carried to next month
    Closure per pixel: P − etrain − SRO − PERC = ΔSM   (blue ET is supplied from outside the bucket).
    Returns dict of DataArrays: etrain, etincr, sro, perc, sm.
    """
    T = P.sizes["time"]
    smmax = np.asarray(SMmax, "float32")
    SM = smmax * sm_init_frac
    out = {k: np.zeros((T,) + smmax.shape, "float32") for k in ["etrain", "etincr", "sro", "perc", "sm"]}
    for t in range(T):
        Pt = np.nan_to_num(np.asarray(P.isel(time=t)))
        Et = np.nan_to_num(np.asarray(ET.isel(time=t)))
        It = np.nan_to_num(np.asarray(I.isel(time=t)))
        nrd = np.maximum(np.nan_to_num(np.asarray(NRD.isel(time=t))), 1.0)
        if et_includes_interception:
            Ei = np.minimum(np.minimum(It, Et), Pt)
            Pe = np.maximum(Pt - Ei, 0.0)
            Es = np.maximum(Et - Ei, 0.0)
        else:
            Ei = np.zeros_like(Pt)
            Pe = np.maximum(Pt - It, 0.0)
            Es = Et
        x = Pe / nrd
        sro = np.where(Pe > 0, (x ** 2) / (x + cf * np.maximum(smmax - SM, 1e-6)) * nrd, 0.0)
        sro = np.minimum(sro, Pe)
        infil = Pe - sro
        green = SM + infil
        etg = np.minimum(Es, green)
        etb = np.maximum(Es - etg, 0.0)
        after = green - etg
        perc = np.maximum(after - smmax, 0.0)
        SM = np.clip(after - perc, 0.0, smmax)
        out["etrain"][t], out["etincr"][t], out["sro"][t], out["perc"][t], out["sm"][t] = Ei + etg, etb, sro, perc, SM
    coords = {"time": P["time"], P.dims[-2]: P[P.dims[-2]], P.dims[-1]: P[P.dims[-1]]}
    dims = ("time", P.dims[-2], P.dims[-1])
    return {k: xr.DataArray(v, dims=dims, coords=coords, name=k) for k, v in out.items()}


def sm_closure(P, r, SMmax, sm_init_frac, mask):
    """Basin-mean closure residual (mm/month) of a run_sm_balance result:
    P − etrain − SRO − PERC − ΔSM, averaged over the period. Should be ~0 (P = the liquid input used)."""
    m = (mask == 1)
    ydim, xdim = P.dims[-2], P.dims[-1]
    sm0 = xr.DataArray(np.asarray(SMmax, "float32") * sm_init_frac, dims=(ydim, xdim),
                       coords={ydim: P[ydim], xdim: P[xdim]})
    dsm = (r["sm"].isel(time=-1) - sm0)
    flux = (P.fillna(0) - r["etrain"] - r["sro"] - r["perc"]).sum("time")
    res = (flux - dsm).where(m)
    return float(res.mean()) / P.sizes["time"]


def temperature_cube(elev, sealevel_monthly_c, dates, lapse=-0.0065):
    """
    Build a monthly air-temperature cube (°C) from an elevation grid and a 12-value
    sea-level monthly cycle, using a lapse rate: T(t,y,x) = T0[month] + lapse·elevation.
    `elev` is a 2-D DataArray on the WA+ grid; `sealevel_monthly_c` is length-12 (Jan..Dec).
    Screening default — swap in ERA5-Land / DHM station temperature where available.
    """
    e = np.asarray(elev, float)
    months = [pd.Timestamp(t).month for t in dates]
    data = np.stack([sealevel_monthly_c[m - 1] + lapse * e for m in months]).astype("float32")
    ydim, xdim = elev.dims[-2], elev.dims[-1]
    return xr.DataArray(data, dims=("time", ydim, xdim),
                        coords={"time": pd.to_datetime(dates), ydim: elev[ydim], xdim: elev[xdim]},
                        name="T")


def snow_module(P, T, ddf=4.0, t_snow=1.0, t_melt=0.0, trans=1.5, swe_init=0.0,
                glacier=None, ddf_ice=7.0):
    """
    Temperature-index (degree-day) snow + glacier model, monthly. Redistributes
    precipitation in TIME (winter snow → spring/summer melt), and — where a glacier mask
    is given — melts exposed ICE once the seasonal snowpack is gone. Glacier melt is net
    mass loss (a negative ΔS): stored ice released as streamflow, the term that props up
    dry-season flow in a glaciated Himalayan basin.

      rain_frac  = clip((T − (t_snow − trans)) / (2·trans), 0, 1)     # linear snow↔rain split
      snowfall   = P·(1 − rain_frac) ;  rain = P·rain_frac
      PDD        = days_in_month · max(T − t_melt, 0)                 # positive degree-days
      snow_melt  = min(SWE + snowfall, ddf · PDD)                     # snow first, capped at pack
      SWE       += snowfall − snow_melt   (carried to next month)
      ice_melt   = ddf_ice · (PDD − snow_melt/ddf)  where glacier AND SWE exhausted   (else 0)
      liquid     = rain + snow_melt + ice_melt                        # what reaches the soil

    ddf ~3–5 (seasonal snow), ddf_ice ~6–9 (bare ice, darker → melts faster). glacier: a 2-D
    bool/0-1 mask of permanent-ice pixels (e.g. lu==glacier class, or elevation > snowline).
    Returns DataArrays: snowfall, rain, snow_melt, ice_melt, melt (snow+ice), swe, liquid.
    """
    T_ = P.sizes["time"]
    shape = np.asarray(P.isel(time=0)).shape
    days = np.array([pd.Timestamp(t).days_in_month for t in P["time"].values], float)
    gl = None if glacier is None else (np.asarray(glacier) > 0)
    keys = ["snowfall", "rain", "snow_melt", "ice_melt", "melt", "swe", "liquid"]
    out = {k: np.zeros((T_,) + shape, "float32") for k in keys}
    SWE = np.full(shape, float(swe_init), "float32")
    for t in range(T_):
        Pt = np.nan_to_num(np.asarray(P.isel(time=t), float))
        Tt = np.nan_to_num(np.asarray(T.isel(time=t), float))
        rain_frac = np.clip((Tt - (t_snow - trans)) / (2.0 * trans), 0.0, 1.0)
        snowfall = Pt * (1.0 - rain_frac)
        rain = Pt * rain_frac
        pdd = days[t] * np.maximum(Tt - t_melt, 0.0)
        avail = SWE + snowfall
        snow_melt = np.minimum(avail, ddf * pdd)
        SWE = np.maximum(avail - snow_melt, 0.0)
        ice_melt = np.zeros(shape, "float32")
        if gl is not None:
            rem_pdd = np.maximum(pdd - snow_melt / ddf, 0.0)          # degree-days left after snow
            ice_melt = np.where(gl & (SWE <= 1e-6), ddf_ice * rem_pdd, 0.0)
        melt = snow_melt + ice_melt
        out["snowfall"][t], out["rain"][t] = snowfall, rain
        out["snow_melt"][t], out["ice_melt"][t], out["melt"][t] = snow_melt, ice_melt, melt
        out["swe"][t], out["liquid"][t] = SWE, rain + melt
    coords = {"time": P["time"], P.dims[-2]: P[P.dims[-2]], P.dims[-1]: P[P.dims[-1]]}
    dims = ("time", P.dims[-2], P.dims[-1])
    return {k: xr.DataArray(v, dims=dims, coords=coords, name=k) for k, v in out.items()}


def estimate_nrd(P, mm_per_wetday=10.0):
    """Estimate monthly number of rain days from monthly P (fallback when no daily data).
    NRD ≈ P / mm_per_wetday, clipped to [1, days_in_month]. Prefer real CHIRPS-daily counts."""
    nrd = (P / mm_per_wetday).clip(min=1)
    days = xr.DataArray([pd.Timestamp(t).days_in_month for t in P["time"].values],
                        dims=["time"], coords={"time": P["time"]})
    nrd = nrd.where(nrd <= days, days)
    nrd.name = "NRD"
    return nrd


# ============================ Lab 5 helpers (WA+ sheets) ============================
# Per WA+ class: managed(True)/natural(False) and beneficial fraction of ET. Edit for Nepal.
LU_FRACTIONS = {
    1: {"managed": False, "beneficial": 0.00},  # glacier/snow
    2: {"managed": False, "beneficial": 0.85},  # protected forest/wetland
    3: {"managed": False, "beneficial": 0.90},  # forest
    4: {"managed": False, "beneficial": 0.50},  # shrub/grass
    5: {"managed": False, "beneficial": 0.00},  # bare/rock
    6: {"managed": True,  "beneficial": 0.80},  # rainfed crop
    7: {"managed": True,  "beneficial": 0.20},  # settlement/urban
    8: {"managed": True,  "beneficial": 0.85},  # irrigated crop
    9: {"managed": True,  "beneficial": 0.00},  # water body (open-water evap)
}


def _annual_mean_vol(da, m, res):
    """Basin volume (MCM), annual mean, of a monthly depth cube (mm) over mask m."""
    factor = res * res * 1e-9                       # mm·pixel -> MCM
    ydim, xdim = da.dims[-2], da.dims[-1]
    s = (da.where(m).sum(dim=[ydim, xdim]) * factor).to_pandas()
    return float(s.groupby(s.index.year).sum().mean())


def sheet_volumes(cubes, mask, res):
    """Annual-mean basin volumes (MCM) for each flux cube in `cubes`."""
    m = (mask == 1)
    return {k: _annual_mean_vol(v, m, res) for k, v in cubes.items()}


def sheet1_resource_base(V, q_obs=None, net_blue=True):
    """Sheet 1 (MCM/yr): P inflow, ET depletion, modelled outflow (SRO+PERC) and the closure term.

    Modelled outflow = SRO + PERC − blue ET (net_blue=True: blue water is consumed from the basin's own
    rivers/groundwater). storage_change_dS = P − ET − outflow = soil-moisture change (+ any model imbalance).
    net_inflow = P - dS  (= ET + outflow: the water that was actually consumed or left the basin).
    q_obs: optional observed annual outflow (MCM/yr, e.g. Kusum gauge) reported alongside the model.
    """
    outflow = V.get("sro", 0.0) + V.get("perc", 0.0) - (V.get("etincr", 0.0) if net_blue else 0.0)
    dS = V["P"] - V["ET"] - outflow
    out = {"gross_inflow_P": V["P"], "ET_depleted": V["ET"], "outflow_est": outflow,
           "storage_change_dS": dS, "closure_term": dS, "net_inflow": V["P"] - dS,
           "ET_green": V.get("etrain"), "ET_blue": V.get("etincr")}
    if q_obs is not None:
        out["outflow_observed"] = float(q_obs)
        out["outflow_bias_pct"] = 100.0 * (outflow - q_obs) / q_obs if q_obs else float("nan")
    return out


def sheet2_et(ET, lu, mask, res, fractions=LU_FRACTIONS, T=None):
    """Sheet 2 (MCM): ET by class, beneficial/non-beneficial and managed/natural partitions.

    Beneficial ET:
      - default (T is None): beneficial = ET x per-class beneficial fraction (LU assumption).
      - physical (T given, a monthly transpiration cube on the same grid, e.g. WaPOR v3 L1-T-D):
        beneficial = transpiration (the productive consumption); non-beneficial = ET - T
        (soil/open-water evaporation + interception). Use T from the SAME product family as ET
        (ET_SOURCE='wapor') so T <= ET holds; per-class beneficial is clipped to [0, class ET].
    """
    m = (mask == 1); byc = {}; ben_byc = {}; ben = man = nat = tot = 0.0
    for cls, fr in fractions.items():
        mc = m & (lu == cls)
        v = _annual_mean_vol(ET, mc, res)
        byc[cls] = v; tot += v
        if T is not None:
            b = min(_annual_mean_vol(T, mc, res), v)      # beneficial = transpiration, capped at class ET
        else:
            b = v * fr["beneficial"]
        ben_byc[cls] = b; ben += b
        man += v if fr["managed"] else 0.0
        nat += 0.0 if fr["managed"] else v
    return {"by_class": byc, "beneficial_by_class": ben_byc, "total": tot, "beneficial": ben,
            "non_beneficial": tot - ben, "managed": man, "natural": nat,
            "beneficial_source": "WaPOR transpiration (physical)" if T is not None else "LU fractions"}


def sheet3_productivity(ndm, ET, lu, mask, res, price=None, classes=(6, 8), units="kg C"):
    """Sheet 3: biomass, consumed water and water productivity over AGRICULTURAL land (default WA+
    classes 6 rainfed + 8 irrigated cropland; classes=None = whole basin).

    ndm is a per-area flux (e.g. MOD17 GPP in kg C m-2 per month), so mass = sum(ndm x pixel area).
    WP_physical = biomass / ET volume  ->  `units` per m3 of ET.  WP_economic = WP_physical x price
    (price per unit of `units`).
    """
    m = (mask == 1)
    if classes is not None:
        m = m & lu.isin(list(classes))
    ydim, xdim = ndm.dims[-2], ndm.dims[-1]
    area = float(res) * float(res)                                   # m2 per pixel
    et_m3 = _annual_mean_vol(ET, m, res) * 1e6
    bio = (ndm.where(m).sum(dim=[ydim, xdim]) * area).to_pandas()   # mass per month
    bio_annual = float(bio.groupby(bio.index.year).sum().mean())
    out = {"biomass_annual": bio_annual, "ET_m3": et_m3,
           "WP_physical": (bio_annual / et_m3) if et_m3 else float("nan"),
           "classes": list(classes) if classes is not None else "all",
           "area_km2": float(m.sum()) * area / 1e6, "units": f"{units} per m3 of ET"}
    if price is not None:
        out["WP_economic"] = out["WP_physical"] * price
    return out


def sheet4_withdrawals(etincr, sro, perc, lu, mask, res, fractions=LU_FRACTIONS,
                       irrigated=(8,), efficiency=0.5):
    """Sheet 4 (MCM/yr): irrigation withdrawals inferred from consumed blue water.

    consumed_blue_ET = incremental (blue) ET on irrigated land (default WA+ class 8).
    withdrawal_est   = consumed / efficiency   (efficiency = consumed fraction of what is diverted;
                       ~0.4-0.6 for surface irrigation — an ASSUMPTION to state in the log).
    recoverable_return = withdrawal - consumed  (return flow from irrigation, reusable downstream).
    Rain-driven runoff from rainfed fields or settlements is NOT a withdrawal and is excluded.
    Also reports where the basin's blue ET occurs, so Sheet 1's blue ET can be reconciled.
    """
    m = (mask == 1)
    irr = m & lu.isin(list(irrigated))
    managed = [c for c, f in fractions.items() if f["managed"]]
    consumed = _annual_mean_vol(etincr, irr, res)
    withdrawal = consumed / efficiency if efficiency else float("nan")
    return {"consumed_blue_ET": consumed, "withdrawal_est": withdrawal,
            "recoverable_return": withdrawal - consumed, "efficiency": efficiency,
            "irrigated_classes": list(irrigated),
            "blue_ET_irrigated": consumed,
            "blue_ET_other_managed": _annual_mean_vol(etincr, m & lu.isin(managed) & ~lu.isin(list(irrigated)), res),
            "blue_ET_natural": _annual_mean_vol(etincr, m & ~lu.isin(managed), res),
            "runoff_from_irrigated": _annual_mean_vol(sro, irr, res) + _annual_mean_vol(perc, irr, res)}


def sheet_consistency(s1, s2):
    """Cross-sheet identities (MCM/yr) — residuals should be ~0. A non-zero 'S2 total ET − S1 ET' usually
    means land-use cells with no WA+ class (NaN / unmapped codes) inside the basin."""
    out = {"S2 total ET − S1 ET": s2["total"] - s1["ET_depleted"],
           "S2 beneficial+nonben − total": (s2["beneficial"] + s2["non_beneficial"]) - s2["total"],
           "S2 managed+natural − total": (s2["managed"] + s2["natural"]) - s2["total"],
           "S2 sum of classes − total": sum(s2["by_class"].values()) - s2["total"]}
    return out


def sheet_diagnostics(s1, s2=None, s4=None, open_water_E=None, tol_dS=5.0, tol_q=15.0):
    """Plausibility checks that CAN fail (unlike the identities). Returns rows (check, value, limit, ok)."""
    rows = []
    P = s1["gross_inflow_P"]
    rows.append(("|closure term| / P (%)", 100 * abs(s1["storage_change_dS"]) / P if P else float("nan"), tol_dS))
    if s1.get("ET_green") is not None and s1.get("ET_blue") is not None and s1["ET_depleted"]:
        rows.append(("model green+blue ET vs satellite ET (%)",
                     100 * abs(s1["ET_green"] + s1["ET_blue"] - s1["ET_depleted"]) / s1["ET_depleted"], 1.0))
    if "outflow_observed" in s1:
        rows.append(("modelled vs observed outflow (%)", abs(s1["outflow_bias_pct"]), tol_q))
    if s4 is not None and s1.get("ET_blue"):
        rows.append(("S4 consumed > S1 blue ET (MCM)", max(0.0, s4["consumed_blue_ET"] - s1["ET_blue"]), 0.5))
        rows.append(("blue ET on natural land (% of blue ET)", 100 * s4["blue_ET_natural"] / s1["ET_blue"], 25.0))
    if s2 is not None and open_water_E is not None:
        wb = s2["by_class"].get(9, 0.0)
        rows.append(("water-body ET vs JRC open water (%)", 100 * abs(wb - open_water_E) / max(open_water_E, 1e-6), 50.0))
    return [{"check": c, "value": round(float(v), 2), "limit": l, "ok": bool(v <= l)} for c, v, l in rows]


# ============================ Lab 6 helpers (synthesis) ============================
def to_mcm(da, mask, res):
    """Monthly basin volume (MCM) series from a monthly depth cube (mm)."""
    return basin_series(da, mask, how="sum") * (res * res * 1e-9)


def outflow_series(sro, perc, mask, res, baseflow_frac=1.0, etincr=None):
    """Modelled basin outflow (MCM/month) = surface runoff + baseflow_frac·percolation
    (− blue ET, if etincr is given: blue water is drawn from the basin's rivers / groundwater).
    Same-month routing; months that would go negative are set to 0. Prefer route_outflow."""
    q = to_mcm(sro, mask, res) + baseflow_frac * to_mcm(perc, mask, res)
    if etincr is not None:
        q = (q - to_mcm(etincr, mask, res)).clip(lower=0)
    q.name = "Q_MCM"
    return q


def route_outflow(sro, perc, mask, res, alpha=0.0, k=1.0, etincr=None):
    """Basin outflow (MCM/month) with a linear groundwater store.

    fast  = (1 − alpha)·SRO                               (leaves in the same month)
    store inflow = alpha·SRO + PERC − blue ET             (blue ET is abstracted from the store;
                                                           the store cannot go below zero)
    slow  = S·(1 − e^(−1/k)) ;  Q = fast + slow           (k = recession constant, months)
    sro/perc/etincr may be (time,y,x) cubes in mm or monthly MCM pandas Series (faster for calibration).
    alpha=0 and etincr=None with a small k approaches same-month routing. Volume is conserved
    except for blue ET that the store cannot supply (reported as attribute 'unmet_blue_MCM').
    """
    _m = lambda x: x if isinstance(x, pd.Series) else to_mcm(x, mask, res)   # cubes or ready MCM series
    q_sro = _m(sro); q_perc = _m(perc)
    q_blue = _m(etincr) if etincr is not None else q_sro * 0.0
    c = 1.0 - np.exp(-1.0 / max(float(k), 1e-6))
    inflow = alpha * q_sro + q_perc
    S = max(float((inflow - q_blue).mean()) * k, 0.0)       # start near steady state
    slow, unmet = [], 0.0
    for v, b in zip(inflow.values, q_blue.values):
        S += v - b
        if S < 0: unmet += -S; S = 0.0
        q = S * c; S -= q; slow.append(q)
    Q = (1 - alpha) * q_sro + pd.Series(slow, index=q_sro.index)
    Q.name = "Q_MCM"; Q.attrs["unmet_blue_MCM"] = unmet
    return Q


def env_flow(regime, standard="2001", frac2001=0.10, frac2026=0.30, eia=None):
    """Environmental flow regime (MCM/month) under a Nepal standard.
    2001: constant floor = frac·min(monthly-mean flow), or EIA figure if higher.
    2026: frac·(each month's mean flow), applied year-round."""
    if standard == "2001":
        val = frac2001 * float(regime.min())
        if eia is not None:
            val = max(val, eia)
        return pd.Series(val, index=regime.index)
    return frac2026 * regime


def utilisable_flow(q_month, standard, committed_annual=0.0,
                    frac2001=0.10, frac2026=0.30, eia=None):
    """Utilisable flow after environmental + committed flows. Returns regime + annual totals."""
    reg = q_month.groupby(q_month.index.month).mean()          # mean monthly regime (MCM/month)
    env = env_flow(reg, standard, frac2001, frac2026, eia)
    committed = committed_annual / 12.0
    util = (reg - env - committed).clip(lower=0)
    return {"regime": reg, "env": env, "utilisable": util,
            "annual_available": float(reg.sum()), "annual_env": float(env.sum()),
            "annual_committed": float(committed_annual), "annual_utilisable": float(util.sum())}


def scenario_available(q_base, P_series, ET_series, dP=0.0, dET=0.0):
    """First-order scenario availability (MCM/month): Q + dP·P − dET·ET (screening level).
    Kept for a quick look; prefer scenario_run() which actually re-runs the balance."""
    return (q_base + dP * P_series - dET * ET_series).clip(lower=0)


def convert_landuse(lu, from_code, to_code, fraction, seed=0):
    """
    Land-use scenario builder: convert a `fraction` (0..1) of pixels of `from_code`
    to `to_code`, at random. Use it to expand irrigation (rainfed 6 -> irrigated 8),
    afforest (shrub 4 -> forest 3), or degrade (forest 3 -> shrub 4). Returns a new map.
    """
    arr = np.asarray(lu).copy()
    idx = np.argwhere(arr == from_code)
    n = int(round(len(idx) * float(fraction)))
    if n > 0:
        rng = np.random.default_rng(seed)
        for j in rng.choice(len(idx), n, replace=False):
            arr[tuple(idx[j])] = to_code
    if hasattr(lu, "dims"):
        r = lu.copy(data=arr.astype("int16")); r.name = "lu"; return r
    return arr.astype("int16")


def scenario_run(P, ET, LAI, lu, mask, res, ndm=None, *, dP=0.0, dT=0.0, dET=0.0,
                 beta_t=0.03, lu_new=None, cf=1.0, standard="2001",
                 committed_annual=0.0, frac2001=0.10, frac2026=0.30, eia=None,
                 baseflow_frac=1.0, T=None, ddf=4.0, t_snow=1.0, t_melt=0.0,
                 glacier=None, ddf_ice=7.0, awc=0.12, sm_init=0.5, int_k=0.2,
                 alpha=None, k_rec=None, net_blue=False, et_includes_interception=True):
    """
    Full scenario: perturb the drivers, RE-RUN the hydroloop and sheets, and return the
    new utilisable flow plus the sheet terms that move. Screening level, but internally
    consistent — every downstream number follows from the same perturbed inputs.

      Rainfall:      P'  = P·(1 + dP)                     (NRD and interception recomputed)
      Temperature:   ET' = ET·(1 + beta_t·dT + dET)       evaporative-demand change, ~3%/°C
      Land use:      lu' = lu_new (SMmax and the class partitions change)

    CAVEAT: ET here is scaled demand, not physically capped by supply; and the soil-bucket
    model does NOT resolve snow/glacier-melt timing — the single biggest gap for a warming
    Himalayan basin. Read a warming run as 'demand and irrigation requirement rise, outflow
    falls', and treat shifted-melt seasonality qualitatively.

    To reproduce Lab 4 exactly, pass its parameters (saved in hydroloop_params.json): awc (scalar or
    the SoilGrids field), cf, sm_init, int_k, and — if the groundwater store was calibrated — alpha,
    k_rec and net_blue=True. Without them the legacy defaults apply (AWC 0.12, same-month routing).
    """
    P2 = P * (1.0 + dP)
    ET2 = (ET * (1.0 + beta_t * dT + dET)).clip(min=0)
    melt_vol = ice_vol = None
    if T is not None:                                  # snow-aware: warming shifts melt timing
        sn = snow_module(P2, T + dT, ddf=ddf, t_snow=t_snow, t_melt=t_melt,
                         glacier=glacier, ddf_ice=ddf_ice)
        Pin = sn["liquid"]                             # rain + snow-melt + ice-melt reaches the soil
        melt_vol = _annual_mean_vol(sn["melt"], (mask == 1), res)
        ice_vol = _annual_mean_vol(sn["ice_melt"], (mask == 1), res)
    else:
        Pin = P2
    NRD2 = estimate_nrd(Pin)
    I2 = interception_from_lai(LAI, Pin, NRD=NRD2, k=int_k)
    lu2 = lu_new if lu_new is not None else lu
    SMmax2 = smmax_from_lu(lu2, awc=awc)
    fx = run_sm_balance(Pin, ET2, I2, NRD2, np.asarray(SMmax2), cf=cf, sm_init_frac=sm_init,
                        et_includes_interception=et_includes_interception)
    if alpha is not None and k_rec is not None:
        Q2 = route_outflow(fx["sro"], fx["perc"], mask, res, alpha=alpha, k=k_rec,
                           etincr=fx["etincr"] if net_blue else None)
    else:
        Q2 = outflow_series(fx["sro"], fx["perc"], mask, res, baseflow_frac,
                            etincr=fx["etincr"] if net_blue else None)
    util = utilisable_flow(Q2, standard, committed_annual, frac2001, frac2026, eia)
    V = sheet_volumes({"P": P2, "ET": ET2, "etrain": fx["etrain"], "etincr": fx["etincr"],
                       "sro": fx["sro"], "perc": fx["perc"]}, mask, res)
    s1 = sheet1_resource_base(V, net_blue=net_blue); s2 = sheet2_et(ET2, lu2, mask, res)
    out = {"annual_utilisable": util["annual_utilisable"], "annual_available": util["annual_available"],
           "annual_env": util["annual_env"], "utilisable": util["utilisable"], "Q": Q2,
           "ET_total": s2["total"], "blue_ET": V.get("etincr"), "outflow": s1["outflow_est"],
           "beneficial": s2["beneficial"], "sheet1": s1, "sheet2": s2}
    if melt_vol is not None:
        out["melt"] = melt_vol
        out["glacier_melt"] = ice_vol
    if ndm is not None:
        out["sheet3"] = sheet3_productivity(ndm, ET2, lu2, mask, res)
    return out


# ============================ Optional: MIKE SHE ingest ============================
def to_wa_cube(da, gridspec, agg="sum", freq="MS", resampling="bilinear"):
    """Resample a (time,y,x) rio-georeferenced DataArray to monthly and onto the WA+ grid.
    agg='sum' for fluxes (ET, runoff, recharge), 'mean' for states (soil moisture, head)."""
    if "time" in da.dims:
        rs = da.resample(time=freq)
        da = rs.sum() if agg == "sum" else rs.mean()
    return regrid(da, gridspec, resampling=resampling)


def load_mike_she(var_map, gridspec, mike_crs, out_dir="."):
    """
    OPTIONAL — ingest MIKE SHE outputs into the WA+ cube. Needs `mikeio` (pip install mikeio;
    `mikeio1d` for river .res1d discharge). Default WA+ runs do NOT use this.

    var_map: {cube_key: {'path': <.dfs2>, 'item': <item name>, 'agg': 'sum'|'mean', 'nc': <file>}}
      e.g. {'et':{'path':'ET.dfs2','item':'ET','agg':'sum'},
            'sro':{'path':'OL.dfs2','item':'overland flow','agg':'sum'},
            'perc':{'path':'recharge.dfs2','item':'recharge','agg':'sum'},
            'i':{'path':'interception.dfs2','item':'interception','agg':'sum'}}
    Returns {cube_key: DataArray} on the WA+ grid and writes each to NetCDF.
    """
    import mikeio
    out = {}
    for key, spec in var_map.items():
        ds = mikeio.read(spec["path"], items=[spec["item"]])
        da = ds[spec["item"]].to_xarray()
        ren = {}
        for d in list(da.dims):
            dl = str(d).lower()
            if dl in ("y", "northing", "lat", "j"): ren[d] = "y"
            if dl in ("x", "easting", "lon", "i"): ren[d] = "x"
        da = da.rename(ren)
        da = da.rio.write_crs(mike_crs)
        cube = to_wa_cube(da, gridspec, agg=spec.get("agg", "sum"))
        cube.name = key
        cube.to_netcdf(os.path.join(out_dir, spec.get("nc", f"{key}.nc")))
        out[key] = cube
    return out


def discharge_to_mcm(q_m3s):
    """Convert an m3/s discharge time series (datetime-indexed) to monthly VOLUME in MCM."""
    idx = q_m3s.index
    dt = idx.to_series().diff().dt.total_seconds()
    dt.iloc[0] = dt.iloc[1] if len(dt) > 1 else 86400.0     # first step = second step
    vol = pd.Series(np.asarray(q_m3s.values, float) * dt.values, index=idx)
    m = vol.resample("MS").sum() / 1e6
    m.name = "Q_MCM"
    return m


def load_mike_river(res1d_path, column=None):
    """
    OPTIONAL — read MIKE Hydro / MIKE 11 river discharge (.res1d) and return a monthly
    discharge VOLUME series (MCM/month) for use as basin outflow in Lab 6. Needs `mikeio1d`.
    column: the discharge column to use; default = the last 'Discharge' column (usually the outlet).
    Discharge is assumed in m3/s.
    """
    from mikeio1d import Res1D
    df = Res1D(res1d_path).read()                 # DataFrame, datetime index, m3/s columns
    if column is None:
        cand = [c for c in df.columns if "Discharge" in str(c)]
        if not cand:
            raise ValueError("No 'Discharge' column found in the .res1d — pass column=<name>.")
        column = cand[-1]
    return discharge_to_mcm(df[column])
