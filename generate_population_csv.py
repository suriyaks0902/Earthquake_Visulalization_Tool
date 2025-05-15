# import pandas as pd
# import numpy as np
# import rasterio
# from shapely.geometry import Point, mapping
# import geopandas as gpd
# from rasterio.mask import mask
# from tqdm import tqdm
# import os

# # === CONFIG ===
# INPUT_CSV = "earthquakes_with_iso3.csv"
# OUTPUT_CSV = "earthquakes_with_population.csv"
# GLOBAL_DIR = r"C:\Users\kssur\SURIYA\Data Visualization\Final_project\earthquake-visualization-tool\global_population_tiles_2024"
# USA_DIR = r"C:\Users\kssur\SURIYA\Data Visualization\Final_project\earthquake-visualization-tool\population_tiles_2024"
# RADIUS_KM = 10

# # === Load input
# df = pd.read_csv(INPUT_CSV)

# # Add column if not already present
# if "state_code" not in df.columns:
#     df["state_code"] = None  # fallback

# # === Cache available tif files
# available_global = {f.split("_")[0] for f in os.listdir(GLOBAL_DIR) if f.endswith(".tif")}
# available_usa = {f.split("_")[0] for f in os.listdir(USA_DIR) if f.endswith(".tif")}
# print(f"[INFO] Found {len(available_global)} global and {len(available_usa)} US state tiles")

# # === Population loader with fallback
# def load_population(lat, lon, iso3, state_code=None):
#     # if iso3 == "USA" and state_code and state_code.lower() in available_usa:
#     if iso3 == "USA" and isinstance(state_code, str) and state_code.lower() in available_usa:
#         tif_path = os.path.join(USA_DIR, f"{state_code.lower()}_pop_2024_CN_100m_R2024A_v1.tif")
#     elif iso3.lower() in available_global:
#         tif_path = os.path.join(GLOBAL_DIR, f"{iso3.lower()}_pop_2024_CN_100m_R2024A_v1.tif")
#     else:
#         return np.nan

#     try:
#         with rasterio.open(tif_path) as src:
#             crs = src.crs
#             gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs(crs)
#             # buffered = gdf.buffer(RADIUS_KM * 1000)
#             gdf_proj = gdf.to_crs("EPSG:3857")  # Project to meters
#             buffered = gdf_proj.buffer(RADIUS_KM * 1000)
#             buffered = buffered.to_crs("EPSG:4326")  # Back to lat/lon for masking
#             out_image, _ = mask(src, buffered.geometry.map(mapping), crop=True)
#             data = out_image[0]
#             data[data < 0] = 0
#             return np.sum(data)
#     except Exception as e:
#         print(f"[ERROR] {tif_path}: {e}")
#         return np.nan

# import geopandas as gpd
# from shapely.geometry import Point

# # === Load US States shapefile (once)
# STATES_SHP = "us_states/cb_2022_us_state_500k.shp"
# states_gdf = gpd.read_file(STATES_SHP).to_crs("EPSG:4326")  # WGS84

# # === Filter USA rows needing state_code
# usa_df = df[(df["country_iso3"] == "USA") & (df["state_code"].isna())].copy()

# # === Convert USA quake points to GeoDataFrame
# geometry = [Point(xy) for xy in zip(usa_df["longitude"], usa_df["latitude"])]
# usa_points = gpd.GeoDataFrame(usa_df, geometry=geometry, crs="EPSG:4326")

# # === Spatial join: assign state FIPS / postal code
# matched = gpd.sjoin(usa_points, states_gdf[["STUSPS", "geometry"]], how="left", predicate='intersects')

# # === Fill back into original dataframe
# df.loc[matched.index, "state_code"] = matched["STUSPS"].values
# print(f"[INFO] Auto-filled state_code for {matched['STUSPS'].notna().sum()} USA earthquakes.")


# # === Estimate population with tqdm progress
# print("[INFO] Estimating population around each earthquake...")
# pops = []
# for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing earthquakes"):
#     pop = load_population(row["latitude"], row["longitude"], row["country_iso3"], row["state_code"])
#     pops.append(pop)

# df["estimated_population"] = pops
# df["log_population"] = df["estimated_population"].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)

# # === Save result
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Done. Saved: {OUTPUT_CSV}")


# For the second part of the code, which calculates distances to fault lines and plate boundaries:

# # === CONFIG ===
# import pandas as pd
# import geopandas as gpd
# from shapely.geometry import Point
# from shapely.ops import nearest_points
# import os

# # === Config paths ===
# CSV_PATH = "earthquakes_with_population.csv"
# FAULT_PATH = "fault_lines/Qfaults_US_Database.shp"
# PLATE_PATH = r"plate_boundaries/PB2002_boundaries.shp"

# # === Check required files
# for path in [CSV_PATH, FAULT_PATH, PLATE_PATH]:
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Missing file: {path}")

# # === Load and reproject data
# print("[INFO] Loading earthquake CSV...")
# df = pd.read_csv(CSV_PATH)
# eq_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326")

# print("[INFO] Loading fault and plate shapefiles...")
# faults_gdf = gpd.read_file(FAULT_PATH).to_crs("EPSG:4326")
# plates_gdf = gpd.read_file(PLATE_PATH).to_crs("EPSG:4326")

# print("[INFO] Reprojecting all to EPSG:3857...")
# eq_gdf = eq_gdf.to_crs(epsg=3857)
# faults_gdf = faults_gdf.to_crs(epsg=3857)
# plates_gdf = plates_gdf.to_crs(epsg=3857)

# # === Distance computation
# def get_nearest_distance_km(point, gdf):
#     nearest_geom = gdf.geometry.unary_union
#     nearest_point = nearest_points(point, nearest_geom)[1]
#     return point.distance(nearest_point) / 1000

# print("[INFO] Calculating fault distances...")
# eq_gdf["fault_distance_km"] = eq_gdf.geometry.apply(lambda x: get_nearest_distance_km(x, faults_gdf))

# print("[INFO] Calculating plate distances...")
# eq_gdf["plate_distance_km"] = eq_gdf.geometry.apply(lambda x: get_nearest_distance_km(x, plates_gdf))

# # === Save to file
# print("[INFO] Saving final CSV...")
# eq_gdf.drop(columns="geometry").to_csv(CSV_PATH, index=False)
# print("✅ Done: 'fault_distance_km' and 'plate_distance_km' added to:", CSV_PATH)



# this code below is to estimate the radius of the earthquake using GMPE and then estimate the population within that radius.


# # === CONFIG ===

# import pandas as pd
# import numpy as np
# import rasterio
# from shapely.geometry import Point, mapping
# import geopandas as gpd
# from rasterio.mask import mask
# from tqdm import tqdm
# import os
# from gmpe_radius_estimator import estimate_radius_pga_gmpe
# # from openquake.hazardlib.imt import PGA
# # from openquake.hazardlib.geo import Point as OQ_Point
# # from openquake.hazardlib.gsim.boore_2014 import BooreEtAl2014
# # from openquake.hazardlib.contexts import SitesContext, RuptureContext, DistancesContext

# # === Paths
# INPUT_CSV = "earthquakes_with_iso3.csv"
# OUTPUT_CSV = "earthquakes_with_population.csv"
# GLOBAL_DIR = "global_population_tiles_2024"
# USA_DIR = "population_tiles_2024"
# STATES_SHP = "us_states/cb_2022_us_state_500k.shp"

# # === Load base earthquake data
# df = pd.read_csv(INPUT_CSV)
# df["state_code"] = df.get("state_code", None)

# # === Load US states and assign missing state_code if needed
# states_gdf = gpd.read_file(STATES_SHP).to_crs("EPSG:4326")
# usa_df = df[(df["country_iso3"] == "USA") & (df["state_code"].isna())].copy()
# geometry = [Point(xy) for xy in zip(usa_df["longitude"], usa_df["latitude"])]
# usa_points = gpd.GeoDataFrame(usa_df, geometry=geometry, crs="EPSG:4326")
# matched = gpd.sjoin(usa_points, states_gdf[["STUSPS", "geometry"]], how="left", predicate='intersects')
# df.loc[matched.index, "state_code"] = matched["STUSPS"].values

# # === Load available population tiles
# available_global = {f.split("_")[0] for f in os.listdir(GLOBAL_DIR) if f.endswith(".tif")}
# available_usa = {f.split("_")[0] for f in os.listdir(USA_DIR) if f.endswith(".tif")}

# # === GMPE radius function

# # === Population estimation within dynamic radius
# def load_population(lat, lon, iso3, state_code, radius_km):
#     if iso3 == "USA" and isinstance(state_code, str) and state_code.lower() in available_usa:
#         tif_path = os.path.join(USA_DIR, f"{state_code.lower()}_pop_2024_CN_100m_R2024A_v1.tif")
#     elif iso3.lower() in available_global:
#         tif_path = os.path.join(GLOBAL_DIR, f"{iso3.lower()}_pop_2024_CN_100m_R2024A_v1.tif")
#     else:
#         return np.nan

#     try:
#         with rasterio.open(tif_path) as src:
#             gdf = gpd.GeoDataFrame(geometry=[Point(lon, lat)], crs="EPSG:4326").to_crs("EPSG:3857")
#             buffered = gdf.buffer(radius_km * 1000).to_crs("EPSG:4326")
#             out_image, _ = mask(src, buffered.geometry.map(mapping), crop=True)
#             data = out_image[0]
#             data[data < 0] = 0
#             return np.sum(data)
#     except Exception as e:
#         print(f"[ERROR] {tif_path}: {e}")
#         return np.nan

# # === Loop and compute
# print("[INFO] Estimating radius and population using GMPE-based spread...")
# radii, populations = [], []

# for _, row in tqdm(df.iterrows(), total=len(df)):
#     lat, lon, mag, iso3, state_code = row["latitude"], row["longitude"], row["mag"], row["country_iso3"], row["state_code"]
#     radius = estimate_radius_pga_gmpe(mag, lat, lon)
#     pop = load_population(lat, lon, iso3, state_code, radius_km=radius)
#     radii.append(radius)
#     populations.append(pop)

# df["estimated_radius_km"] = radii
# df["estimated_population"] = populations
# df["log_population"] = df["estimated_population"].apply(lambda x: np.log1p(x) if pd.notna(x) else np.nan)

# # === Save
# df.to_csv(OUTPUT_CSV, index=False)
# print(f"\n✅ Saved updated dataset: {OUTPUT_CSV}")


############ extracting total population from the population tiles

import os
import rasterio
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm

GLOBAL_DIR = r"C:\Users\kssur\SURIYA\Data Visualization\Final_project\earthquake-visualization-tool\global_population_tiles_2024"
USA_DIR = r"C:\Users\kssur\SURIYA\Data Visualization\Final_project\earthquake-visualization-tool\population_tiles_2024"
OUTPUT_CSV = "total_population_by_country_2024.csv"

def extract_iso_from_filename(filepath):
    """
    Extract ISO3 code from filename, assuming pattern .../<ISO3>_ppp_2024.tif
    or for US states: .../<USA_XX>_ppp_2024.tif (returns 'USA')
    """
    filename = os.path.basename(filepath)
    if filename.startswith("USA_"):
        return "USA"
    else:
        return filename.split("_")[0]

def sum_population_from_tifs(directory):
    """
    Returns a dictionary: {ISO3: total_population} from all GeoTIFFs in a directory
    """
    pop_totals = {}
    for tif in tqdm(glob(os.path.join(directory, "*.tif")), desc=f"Processing {directory}"):
        iso3 = extract_iso_from_filename(tif)
        try:
            with rasterio.open(tif) as src:
                data = src.read(1)
                data[data < 0] = 0  # Clean invalid
                total = np.nansum(data)

                # Accumulate population per country
                if iso3 in pop_totals:
                    pop_totals[iso3] += total
                else:
                    pop_totals[iso3] = total
        except Exception as e:
            print(f"Error reading {tif}: {e}")
    return pop_totals

# === Process global and US tiles ===
global_pop = sum_population_from_tifs(GLOBAL_DIR)
usa_pop = sum_population_from_tifs(USA_DIR)

# === Merge and save ===
combined = global_pop
for k, v in usa_pop.items():
    combined[k] = combined.get(k, 0) + v

df = pd.DataFrame(list(combined.items()), columns=["country_iso3", "total_population"])
df["total_population"] = df["total_population"].astype(int)
df = df.sort_values("total_population", ascending=False)
df.to_csv(OUTPUT_CSV, index=False)

print(f"✅ Saved population totals to: {OUTPUT_CSV}")
