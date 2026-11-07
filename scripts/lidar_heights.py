#!/usr/bin/env python3
"""Per-building heights for the bundled OSM footprints from the NOAA 2015 Anchorage lidar.

For every OSM footprint:
  roof   = 90th percentile of the first-return surface (1 m cells) inside the footprint shrunk by 1 m
  ground = median of ground-classified cells in a ring 1.5-10 m outside the footprint
  height = roof - ground
Footprints whose height comes out below 2 m are listed as "flat": the lidar saw only ground there,
so the building is probably newer than the 2015 survey (or is a carport/slab) and keeps its default height.

Runs in GitHub Actions (network access to NOAA's public S3 bucket). LIDAR_DIR=... reuses local .laz files.
"""
import json, os, sys, time, urllib.request
import numpy as np
import laspy
import shapely
from shapely.geometry import Polygon
from pyproj import CRS, Transformer

BASE = 'https://noaa-nos-coastal-lidar-pds.s3.amazonaws.com/laz/geoid12b/8406/'
TILES = ['20150510_AK_Anchorage_2015_%s.copc.laz' % t for t in
         ['63436785', '63446784', '63446785', '63446786', '63456784', '63456785', '63456786']]
AOI = (666100.0, 6784800.0, 669100.0, 6787300.0)     # UTM 5N metres: drive-test extent + margin
CELL = 1.0
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, 'lidar')
NOISE = {7, 18}
GROUND = 2


def log(*a):
    print(*a, flush=True)


def fetch(name, d):
    p = os.path.join(d, name)
    if os.path.exists(p) and os.path.getsize(p) > 0:
        return p
    url = BASE + name
    for k in range(3):
        try:
            t = time.time()
            urllib.request.urlretrieve(url, p + '.part')
            os.replace(p + '.part', p)
            log(f'  downloaded {name} {os.path.getsize(p) / 1e6:.1f} MB in {time.time() - t:.0f} s')
            return p
        except Exception as e:  # noqa: BLE001
            log(f'  retry {k + 1} {name}: {e}')
            time.sleep(5)
    raise RuntimeError('download failed: ' + name)


def horizontal_crs(las):
    try:
        c = las.header.parse_crs()
    except Exception:  # noqa: BLE001
        c = None
    if c is None:
        return CRS.from_epsg(6334), 'EPSG:6334 (assumed: NAD83(2011) / UTM 5N)'
    if c.is_compound:
        c = c.sub_crs_list[0]
    return c, c.to_string()


def main():
    d = os.environ.get('LIDAR_DIR') or os.path.join(REPO, '.lidar_cache')
    os.makedirs(d, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    x0, y0, x1, y1 = AOI
    nx, ny = int((x1 - x0) / CELL), int((y1 - y0) / CELL)
    dsm = np.full(nx * ny, -np.inf, dtype=np.float64)
    gsum = np.zeros(nx * ny); gcnt = np.zeros(nx * ny)
    crs = crs_txt = None
    npts = 0
    for name in TILES:
        las = laspy.read(fetch(name, d))
        if crs is None:
            crs, crs_txt = horizontal_crs(las)
            log('CRS:', crs_txt)
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        cls = np.asarray(las.classification)
        first = np.asarray(las.return_number) == 1
        m = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        ix = ((x[m] - x0) / CELL).astype(np.int64); iy = ((y[m] - y0) / CELL).astype(np.int64)
        idx = iy * nx + ix
        zc, cc, fc = z[m], cls[m], first[m]
        npts += int(m.sum())
        s = fc & ~np.isin(cc, list(NOISE))                       # surface: first returns, no noise
        np.maximum.at(dsm, idx[s], zc[s])
        g = cc == GROUND
        gsum += np.bincount(idx[g], weights=zc[g], minlength=nx * ny)
        gcnt += np.bincount(idx[g], minlength=nx * ny)
        log(f'{name}: {m.sum():,} points in area, {g.sum():,} ground')
    dsm[~np.isfinite(dsm)] = np.nan
    gnd = np.where(gcnt > 0, gsum / np.maximum(gcnt, 1), np.nan)
    dsm = dsm.reshape(ny, nx); gnd = gnd.reshape(ny, nx)
    log(f'cells with surface {np.isfinite(dsm).mean():.1%}, with ground {np.isfinite(gnd).mean():.1%}; {npts:,} points')

    src = json.load(open(os.path.join(REPO, 'buildings_osm.json'), encoding='utf-8'))
    tf = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
    heights, flat, skipped = {}, [], 0
    for b in src['buildings']:
        xy = [tf.transform(lon, lat) for lon, lat in b['ll']]
        if len(xy) < 4:
            continue
        poly = Polygon(xy)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty:
            continue
        bx0, by0, bx1, by1 = poly.bounds
        if bx1 < x0 + 12 or bx0 > x1 - 12 or by1 < y0 + 12 or by0 > y1 - 12:
            continue
        inner = poly.buffer(-1.0)
        if inner.is_empty or inner.area < 4:
            inner = poly.buffer(-0.3) if not poly.buffer(-0.3).is_empty else poly
        ring = poly.buffer(10).difference(poly.buffer(1.5))

        def cells(geom, grid):
            gx0, gy0, gx1, gy1 = geom.bounds
            i0, i1 = max(0, int((gx0 - x0) / CELL)), min(nx - 1, int((gx1 - x0) / CELL))
            j0, j1 = max(0, int((gy0 - y0) / CELL)), min(ny - 1, int((gy1 - y0) / CELL))
            if i1 < i0 or j1 < j0:
                return np.array([])
            II, JJ = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1))
            cx, cy = x0 + (II + .5) * CELL, y0 + (JJ + .5) * CELL
            inside = shapely.contains_xy(geom, cx, cy)
            return grid[JJ[inside], II[inside]]

        roof = cells(inner, dsm); roof = roof[np.isfinite(roof)]
        gr = cells(ring, gnd); gr = gr[np.isfinite(gr)]
        if roof.size < 3 or gr.size < 3:
            skipped += 1
            continue
        g0 = float(np.median(gr))
        h90 = float(np.percentile(roof, 90)) - g0
        h50 = float(np.percentile(roof, 50)) - g0
        if h90 < 2.0:
            flat.append(b['id'])
            continue
        heights[b['id']] = [round(h90, 1), round(h50, 1), int(roof.size)]

    hs = np.array([v[0] for v in heights.values()]) if heights else np.array([0.0])
    stats = {'count': len(heights), 'flat': len(flat), 'skipped_no_data': skipped,
             'p10': round(float(np.percentile(hs, 10)), 1), 'p25': round(float(np.percentile(hs, 25)), 1),
             'median': round(float(np.median(hs)), 1), 'p75': round(float(np.percentile(hs, 75)), 1),
             'p90': round(float(np.percentile(hs, 90)), 1), 'max': round(float(hs.max()), 1)}
    out = {
        'source': 'NOAA OCM, 2015 Lidar: Municipality of Anchorage, AK (Digital Coast ID 8406), collected May 2015',
        'url': BASE, 'tiles': TILES, 'crs': crs_txt, 'aoi_utm': AOI,
        'method': 'height = p90 of 1 m first-return surface inside footprint shrunk 1 m, minus median ground (class 2) in a 1.5-10 m ring; < 2 m listed as flat',
        'stats': stats, 'heights': heights, 'flat': flat,
    }
    with open(os.path.join(OUT_DIR, 'heights_anchorage.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, separators=(',', ':'))
    log('stats', json.dumps(stats))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        sys.exit(1)
