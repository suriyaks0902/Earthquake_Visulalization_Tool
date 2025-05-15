import os
import requests
import zipfile
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from tqdm import tqdm

# === CONFIG ===
# CSV_PATH = "earthquakes_with_iso3.csv"  # must have 'latitude', 'longitude', 'country_iso3'
# SHAPEFILE_ZIP = "cb_2022_us_state_500k.zip"
# SHAPEFILE_URL = f"https://www2.census.gov/geo/tiger/GENZ2022/shp/{SHAPEFILE_ZIP}"
# SHAPEFILE_DIR = "us_states"
# TIF_URL_PATTERN = (
#     "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024A/2024/"
#     "USA_States/{state}/v1/100m/constrained/"
#     "{state_lower}_pop_2024_CN_100m_R2024A_v1.tif"
# )
# OUTPUT_DIR = "population_tiles_2024"

CSV_PATH = "earthquakes_with_iso3.csv"
OUTPUT_DIR = "global_population_tiles_2024"
BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024A/2024/{iso3}/constrained/{iso3_lower}_ppp_2024_constrained.tif"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === STEP 1: Load and filter USA earthquakes ===
df = pd.read_csv(CSV_PATH)
df_usa = df[df['country_iso3'] == 'USA'].copy()
geometry = [Point(xy) for xy in zip(df_usa.longitude, df_usa.latitude)]
gdf = gpd.GeoDataFrame(df_usa, geometry=geometry, crs="EPSG:4326")

# === STEP 2: Download and extract US state shapefile ===
if not os.path.exists(SHAPEFILE_ZIP):
    print("[INFO] Downloading US state boundaries...")
    with requests.get(SHAPEFILE_URL, stream=True) as r:
        with open(SHAPEFILE_ZIP, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

if not os.path.exists(os.path.join(SHAPEFILE_DIR, "cb_2022_us_state_500k.shp")):
    print("[INFO] Extracting shapefile...")
    with zipfile.ZipFile(SHAPEFILE_ZIP, 'r') as zip_ref:
        zip_ref.extractall(SHAPEFILE_DIR)

shapefile_path = os.path.abspath(os.path.join(SHAPEFILE_DIR, "cb_2022_us_state_500k.shp"))
states_gdf = gpd.read_file(shapefile_path).to_crs("EPSG:4326")

# === STEP 3: Spatial join to get state codes ===
joined = gpd.sjoin(gdf, states_gdf, how="left", predicate="within")
state_codes = sorted(joined['STUSPS'].dropna().unique().tolist())

print(f"\nEarthquakes found in {len(state_codes)} U.S. states:")
print(", ".join(state_codes), "\n")

# === STEP 4: Download population .tif tiles per state ===
def download_with_progress(url, save_path):
    try:
        with requests.get(url, stream=True) as r:
            if r.status_code != 200:
                print(f" {os.path.basename(save_path)} not found at {url}")
                return False
            total = int(r.headers.get('content-length', 0))
            with open(save_path, 'wb') as f, tqdm(
                desc=os.path.basename(save_path),
                total=total,
                unit='B', unit_scale=True, unit_divisor=1024,
                ascii=True
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False

# Loop over states and download
for state in state_codes:
    state_lower = state.lower()
    tif_name = f"{state_lower}_pop_2024_CN_100m_R2024A_v1.tif"
    tif_url = TIF_URL_PATTERN.format(state=state, state_lower=state_lower)
    save_path = os.path.join(OUTPUT_DIR, tif_name)

    if os.path.exists(save_path):
        print(f"[SKIP] Already exists: {tif_name}")
        continue

    download_with_progress(tif_url, save_path)

print("\nAll required state-level population files downloaded.")



import os
import requests
import pandas as pd
from tqdm import tqdm

# === CONFIG ===
CSV_PATH = "earthquakes_with_iso3.csv"
SAVE_DIR = "global_population_tiles_2024"
URL_TEMPLATE = (
    "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2024A/2024/"
    "{ISO3}/v1/100m/constrained/{iso3_lower}_pop_2024_CN_100m_R2024A_v1.tif"
)
SKIP_CODES = {"USA", "USA_States"}

os.makedirs(SAVE_DIR, exist_ok=True)

# === STEP 1: Load country ISO3 codes from earthquake data
df = pd.read_csv(CSV_PATH)
iso3_codes = sorted(set(df['country_iso3'].dropna().unique()))
iso3_codes = [c for c in iso3_codes if c not in SKIP_CODES]  # Skip USA and USA_States

print(f"[INFO] Downloading WorldPop 2024 tiles for {len(iso3_codes)} countries:")
print("→", ", ".join(iso3_codes))

# === STEP 2: Download with progress
def download_with_progress(url, save_path):
    try:
        with requests.get(url, stream=True, timeout=10) as r:
            if r.status_code != 200:
                print(f"Not found: {url}")
                return False
            total = int(r.headers.get("content-length", 0))
            with open(save_path, "wb") as f, tqdm(
                desc=os.path.basename(save_path),
                total=total,
                unit='B', unit_scale=True, unit_divisor=1024,
                ascii=True
            ) as bar:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"[ERROR] {url}: {e}")
        return False

# === STEP 3: Loop and download each valid tile
for ISO3 in iso3_codes:
    iso3_lower = ISO3.lower()
    filename = f"{iso3_lower}_pop_2024_CN_100m_R2024A_v1.tif"
    save_path = os.path.join(SAVE_DIR, filename)
    url = URL_TEMPLATE.format(ISO3=ISO3, iso3_lower=iso3_lower)

    if os.path.exists(save_path):
        print(f"[SKIP] Already downloaded: {filename}")
        continue

    print(f"[↓] Downloading {filename} ...")
    download_with_progress(url, save_path)

print("\nDone! All available tiles downloaded successfully.")
