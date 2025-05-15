import os
import requests
from zipfile import ZipFile
import rasterio
from rasterio.windows import from_bounds

SRTM_CACHE = "data/srtm_tiles"
SRTM_BASE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/skadi"

def get_tile_name(lat, lon):
    lat_prefix = 'N' if lat >= 0 else 'S'
    lon_prefix = 'E' if lon >= 0 else 'W'
    lat_str = f"{lat_prefix}{int(abs(lat)):02d}"
    lon_str = f"{lon_prefix}{int(abs(lon)):03d}"
    return f"{lat_str}{lon_str}"

def download_srtm_tile(tile_name, cache_dir="data/srtm_tiles"):
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, f"{tile_name}.hgt.zip")
    hgt_path = zip_path.replace('.zip', '')

    # ✅ Already downloaded?
    if os.path.exists(hgt_path):
        return hgt_path

    # ✅ Try known public mirror
    url = f"https://dwtkns.com/srtm30m/{tile_name}.hgt.zip"
    print(f"Downloading {tile_name} from {url}...")

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)

            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)

            return hgt_path
    except Exception as e:
        print(f"Download failed: {e}")

    print(f"❌ Tile {tile_name} not available from primary source.")
    return None  # let app fallback to synthetic

def load_local_terrain(lat, lon, tile_path, size_deg=1.0):
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(tile_path) as src:
        half = size_deg / 2
        try:
            window = from_bounds(
                lon - half, lat - half,
                lon + half, lat + half,
                src.transform
            )
            elevation = src.read(1, window=window)
            if elevation.size == 0 or np.isnan(elevation).all():
                print("⚠️ Empty or invalid elevation data.")
                return None, None, None
            bounds = src.window_bounds(window)
            transform = src.window_transform(window)
            print(f"[DEBUG] elevation.shape = {elevation.shape}")
            print(f"[DEBUG] elevation min: {elevation.min()}, max: {elevation.max()}, nan count: {np.isnan(elevation).sum()}")
            return elevation, bounds, transform
        except Exception as e:
            print(f"❌ Failed to load terrain: {e}")
            return None, None, None


from rasterio.transform import xy
import numpy as np

def generate_grid_from_transform(transform, shape):
    """
    Given a rasterio transform and shape (rows, cols), generate meshgrid of lat/lon coordinates.
    """
    nrows, ncols = shape
    xs = np.zeros(ncols)
    ys = np.zeros(nrows)

    for col in range(ncols):
        xs[col], _ = xy(transform, 0, col)
    for row in range(nrows):
        _, ys[row] = xy(transform, row, 0)

    xx, yy = np.meshgrid(xs, ys)
    return xx, yy
