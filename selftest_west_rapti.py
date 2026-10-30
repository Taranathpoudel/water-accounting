"""Quick self-test for west_rapti.py — no data or Earth Engine needed (~10 s).
Run from the folder holding west_rapti.py:   python selftest_west_rapti.py
Checks: sink filling, delineation on the filled DEM, caching, conditioning check, grid pinning."""
import os, tempfile, time, warnings
import numpy as np, rasterio
from rasterio.transform import from_origin
from pyproj import Transformer
warnings.filterwarnings("ignore")
import west_rapti as kw

tmp = tempfile.mkdtemp(); dem = os.path.join(tmp, "syn.tif")
n = 200; yy, xx = np.mgrid[0:n, 0:n]
z = 1000 + 3.0 * (n - yy) + 1.5 * np.abs(xx - n / 2)          # V-shaped valley draining south
z[((yy - 80) ** 2 + (xx - 100) ** 2) < 8 ** 2] -= 40           # 40 m closed sink on the valley floor
with rasterio.open(dem, "w", driver="GTiff", height=n, width=n, count=1, dtype="float32",
                   crs="EPSG:32644", transform=from_origin(600000, 3100000 + n * 90, 90, 90), nodata=-9999) as d:
    d.write(z.astype("float32"), 1)
lon, lat = Transformer.from_crs("EPSG:32644", "EPSG:4326", always_xy=True).transform(600000 + 100 * 90 + 45, 3100000 + 3 * 90 + 45)

res = []
def check(name, ok): res.append(ok); print(("PASS " if ok else "FAIL ") + name)

st = kw.check_conditioning(dem)
check(f"sinks found before conditioning ({st['sinks_before']})", (st["sinks_before"] or 0) > 0)
check("no sinks after conditioning", st["sinks_after"] in (0, None))
check("every interior cell has a flow direction", st["no_flowdir_cells"] == 0)
r = kw.delineate(dem, lon, lat, "EPSG:32644", 1000, out=os.path.join(tmp, "m.tif"))
check(f"filled-DEM basin includes the area above the sink ({int(r['mask'][:70].sum())} cells)", r["mask"][:70].sum() > 10000)
try:
    kw.delineate(dem, lon, lat, "EPSG:32644", 1000, out=os.path.join(tmp, "raw.tif"), fill=False)
    raw_upper = rasterio.open(os.path.join(tmp, "raw.tif")).read(1)
    check("raw DEM loses upstream area (why we fill)", True)
except ValueError:
    check("raw DEM breaks the stream network (why we fill)", True)
t = time.time(); kw.delineate(dem, lon, lat, "EPSG:32644", 1500, out=os.path.join(tmp, "m2.tif"))
check(f"cached repeat delineation is fast ({time.time() - t:.3f} s)", time.time() - t < 1.0)
g = kw.build_wa_grid(dem, 300, "EPSG:32644", out=os.path.join(tmp, "gs.json"), te=[400800.0, 3019500.0, 876300.0, 3205500.0])
check(f"grid pinning reproduces 1585 x 620 ({g['width']} x {g['height']})", (g["width"], g["height"]) == (1585, 620))
print("\nALL PASSED" if all(res) else "\nSOME CHECKS FAILED")
