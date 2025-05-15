# import os
# from tqdm import tqdm
# import requests
# from zipfile import ZipFile, BadZipFile
# from srtm_downloader import download_srtm_tile_nasa

# # 📍 Parse tile name to lat/lon
# def parse_tile_name(name):
#     lat = int(name[1:3]) * (1 if name[0] == 'N' else -1)
#     lon = int(name[5:8]) * (1 if name[4] == 'E' else -1)
#     return lat, lon

# # 🔁 Format back to SRTM tile name
# def format_tile_name(lat, lon):
#     lat_prefix = 'N' if lat >= 0 else 'S'
#     lon_prefix = 'E' if lon >= 0 else 'W'
#     return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"

# # 🔄 Generate all tile names in range
# def generate_tile_range(start_name, end_name):
#     def parse_tile_name(name):
#         lat_hem = name[0]
#         lat = int(name[1:3]) * (1 if lat_hem == 'N' else -1)
#         lon_hem = name[3]
#         lon = int(name[4:7]) * (1 if lon_hem == 'E' else -1)
#         return lat, lon

#     def format_tile_name(lat, lon):
#         lat_prefix = 'N' if lat >= 0 else 'S'
#         lon_prefix = 'E' if lon >= 0 else 'W'
#         return f"{lat_prefix}{abs(lat):02d}{lon_prefix}{abs(lon):03d}"

#     start_lat, start_lon = parse_tile_name(start_name)
#     end_lat, end_lon = parse_tile_name(end_name)

#     # Ensure we iterate in increasing order
#     lat_range = range(min(start_lat, end_lat), max(start_lat, end_lat) + 1)
#     lon_range = range(min(start_lon, end_lon), max(start_lon, end_lon) + 1)

#     tile_names = []
#     for lat in lat_range:
#         for lon in lon_range:
#             tile_names.append(format_tile_name(lat, lon))

#     return tile_names



# # 📦 Your defined page ranges
# page_ranges = {
#     "Page_1": ("N00E006", "N21E042"),
#     "Page_2": ("N21E043", "N36W114"),
#     "Page_3": ("N36W114", "N50E007"),
#     "Page_4": ("N50E007", "S02E113"),
#     "Page_5": ("S02E113", "S24E027"),
#     "Page_6": ("S24E028", "S56W072"),
# }

# # 🔽 Download all tiles for all pages
# failed_tiles = []
# for page, (start_tile, end_tile) in page_ranges.items():
#     print(f"\n🗂 Downloading tiles for {page}: {start_tile} → {end_tile}")
#     tile_list = generate_tile_range(start_tile, end_tile)
#     for tile in tile_list:
#         try:
#             download_srtm_tile_nasa(tile)
#         except Exception as e:
#             print(f"❌ Failed to download {tile}: {e}")
#             failed_tiles.append(tile)
# # for page, (start_tile, end_tile) in page_ranges.items():
# #     print(f"\n🗂 Downloading tiles for {page}: {start_tile} → {end_tile}")
# #     tile_list = generate_tile_range(start_tile, end_tile)
# #     for tile in tile_list:
# #         download_srtm_tile_nasa(tile)
# if failed_tiles:
#     with open("failed_tiles.log", "w") as f:
#         for tile in failed_tiles:
#             f.write(f"{tile}\n")
#     print("❌ Some tiles failed. See failed_tiles.log for details.")

#     # print(generate_tile_range("N00E006", "N21E042")[:10])

#     # print(f"🧪 Tiles for {page}:")
#     # print(tile_list[:10])

#     # tile_list = generate_tile_range(start_tile, end_tile)
#     # print("Tiles:", tile_list)




# # import netrc

# # try:
# #     creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
# #     print("✅ netrc credentials found:", creds)
# # except FileNotFoundError:
# #     print("❌ .netrc file not found.")
# # except Exception as e:
# #     print("❌ Error reading .netrc:", e)



### DOWNLOAD SRTM TILES ###
# import os
# import pandas as pd
# from tqdm import tqdm
# from srtm_downloader import download_srtm_tile_nasa
# from terrain_loader import get_tile_name  # This should already exist

# # 📥 Load earthquake dataset
# DATA_PATH = "earthquakes.csv"

# if not os.path.exists(DATA_PATH):
#     print(f"❌ Dataset not found at {DATA_PATH}")
#     exit()

# df = pd.read_csv(DATA_PATH)

# # 🎯 Extract unique SRTM tile names from quake locations
# unique_tiles = set()
# for _, row in df.iterrows():
#     lat, lon = row['latitude'], row['longitude']
#     tile = get_tile_name(lat, lon)
#     unique_tiles.add(tile)

# tile_list = sorted(unique_tiles)

# # 💾 Estimate total size
# tile_count = len(tile_list)
# estimated_size_mb = tile_count * 2.8  # Each tile ≈ 2.5 MB

# print(f"\n📊 Total unique SRTM tiles to download: {tile_count}")
# estimated_size_gb = estimated_size_mb / 1024
# print(f"💾 Estimated storage required: ~{estimated_size_mb:.2f} MB ({estimated_size_gb:.2f} GB)")


# # 🤔 Ask user
# proceed = input("\n✅ Proceed with download? (yes/no): ").strip().lower()
# if proceed != "yes":
#     print("❌ Download cancelled by user.")
#     exit()

# # 🚀 Begin download
# failed_tiles = []
# print(f"\n📦 Downloading {tile_count} tiles...\n")

# for tile in tqdm(tile_list, desc="Downloading Tiles"):
#     try:
#         result = download_srtm_tile_nasa(tile)
#         if not result:
#             failed_tiles.append(tile)
#     except Exception as e:
#         print(f"❌ Error downloading {tile}: {e}")
#         failed_tiles.append(tile)

# # 📝 Log any failed downloads
# if failed_tiles:
#     with open("failed_tiles.log", "w") as f:
#         for tile in failed_tiles:
#             f.write(f"{tile}\n")
#     print(f"\n⚠️ {len(failed_tiles)} tile(s) failed. Logged to failed_tiles.log.")
# else:
#     print("\n✅ All tiles downloaded successfully.")



# ## Downlaod WORLDPOP Dataset
# import pandas as pd
# import reverse_geocoder as rg
# import requests
# import os

# # ------------------------------
# # Constants
# # ------------------------------
# CSV_PATH = "earthquakes.csv"
# SAVE_PATH = "population_tiles"
# BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020"

# # ------------------------------
# # Load coordinates in chunks to avoid memory issues
# # ------------------------------
# def load_coordinates(filepath, chunk_size=50000):
#     coords = []
#     for chunk in pd.read_csv(filepath, usecols=["latitude", "longitude"], chunksize=chunk_size):
#         coords.extend(list(zip(chunk["latitude"], chunk["longitude"])))
#     return coords

# # ------------------------------
# # Download GeoTIFF for given 3-letter ISO country code
# # ------------------------------
# def download_population_tile(iso3):
#     country_url = f"{BASE_URL}/{iso3}/"
#     try:
#         r = requests.get(country_url)
#         if r.status_code == 200:
#             for line in r.text.splitlines():
#                 if ".tif" in line and "ppp" in line:
#                     start = line.find('href="') + 6
#                     end = line.find('.tif') + 4
#                     tif_name = line[start:end]
#                     tif_url = f"{country_url}{tif_name}"
#                     out_path = os.path.join(SAVE_PATH, tif_name)

#                     print(f"[DOWNLOAD] {tif_url}")
#                     with requests.get(tif_url, stream=True) as f:
#                         with open(out_path, 'wb') as file:
#                             for chunk in f.iter_content(1024):
#                                 file.write(chunk)
#                     print(f"[SAVED] {out_path}")
#                     return
#         else:
#             print(f"[WARNING] Could not access {country_url} (status {r.status_code})")
#     except Exception as e:
#         print(f"[ERROR] {e} while downloading {iso3}")

# # ------------------------------
# # Main logic wrapped safely
# # ------------------------------
# def main():
#     print("[INFO] Reverse geocoding earthquake coordinates...")
#     coords = load_coordinates(CSV_PATH)

#     # Reverse geocode to get 2-letter ISO country codes
#     locations = rg.search(coords, verbose=True)
#     iso2_codes = sorted(set(loc["cc"] for loc in locations))

#     # Convert to 3-letter ISO using static mapping (ISO 3166-1 alpha-3)
#     import pycountry
#     iso3_codes = []
#     for code in iso2_codes:
#         try:
#             iso3 = pycountry.countries.get(alpha_2=code).alpha_3
#             iso3_codes.append(iso3)
#         except:
#             print(f"[SKIP] No ISO-3 match for: {code}")

#     # Create output folder
#     os.makedirs(SAVE_PATH, exist_ok=True)

#     # Download tiles
#     for iso3 in sorted(set(iso3_codes)):
#         print(f"[INFO] Checking population data for {iso3}...")
#         download_population_tile(iso3)

# # ------------------------------
# # Entry point for Windows multiprocessing
# # ------------------------------
# if __name__ == "__main__":
#     main()
#     print("[INFO] Population data download complete.")



# import pandas as pd
# import pycountry

# # === Load your earthquake CSV file
# CSV_PATH = "earthquakes_with_tiles.csv"
# df = pd.read_csv(CSV_PATH)

# # === Extract country names from the 'place' column
# df['country'] = df['place'].str.split(',').str[-1].str.strip()

# # === Convert country names to ISO3 codes using pycountry
# def to_iso3(name):
#     try:
#         return pycountry.countries.lookup(name).alpha_3
#     except:
#         return None

# df['iso3'] = df['country'].apply(to_iso3)

# # === Drop missing or unknown mappings
# iso3_codes = sorted(df['iso3'].dropna().unique())

# print(f"\n[INFO] Found {len(iso3_codes)} unique ISO3 country codes:\n")
# print(iso3_codes)

# # === Print the WorldPop GeoTIFF folder URLs for each country
# print("\n[RESULT] Download folders (inspect for .tif files):\n")
# BASE_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020/2020"

# for code in iso3_codes:
#     print(f"{code}: {BASE_URL}/{code}/")


import pandas as pd
import pycountry
import requests
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import time

# === Step 1: Load earthquake coordinates
print("[STEP] Reading 'earthquakes.csv'...")
df = pd.read_csv("earthquakes.csv")

# === Ensure country column exists using reverse geocoding
if 'country' not in df.columns:
    print("[INFO] 'country' column not found. Running reverse geocoding...")
    geolocator = Nominatim(user_agent="eq_downloader")

    def get_country_code(lat, lon, retries=3):
        for _ in range(retries):
            try:
                loc = geolocator.reverse((lat, lon), language="en", timeout=10)
                if loc and 'country_code' in loc.raw['address']:
                    return loc.raw['address']['country_code'].upper()
            except GeocoderTimedOut:
                time.sleep(1)
            except Exception as e:
                print(f"[ERROR] Geocoding failed for ({lat}, {lon}): {e}")
        return None

    df['country'] = df.apply(lambda row: get_country_code(row['latitude'], row['longitude']), axis=1)
    df.to_csv("earthquakes_with_country.csv", index=False)
    print("[INFO] Reverse geocoding complete. Saved to 'earthquakes_with_country.csv'.")

# === Step 2: Extract ISO2 codes
iso2_codes = df['country'].dropna().unique()
print("[DEBUG] Unique ISO2 codes extracted from earthquake dataset:", iso2_codes)
print(f"[INFO] Found {len(iso2_codes)} unique ISO2 country codes: {list(iso2_codes)}\n")

# === Step 3: Convert ISO2 to ISO3
def iso2_to_iso3(iso2):
    try:
        return pycountry.countries.get(alpha_2=iso2.upper()).alpha_3
    except Exception as e:
        print(f"[WARN] Could not convert ISO2 '{iso2}': {e}")
        return None

iso3_codes = []
for code in iso2_codes:
    iso3 = iso2_to_iso3(code)
    if iso3:
        print(f"[OK] Converted {code} ➝ {iso3}")
        iso3_codes.append(iso3)
    else:
        print(f"[SKIP] Skipped {code} (invalid)")

iso3_codes = sorted(set(iso3_codes))
print(f"\n[STEP] Unique ISO3 codes to check: {iso3_codes}\n")

# === Step 4: Check WorldPop download links
print("[STEP] Checking available download URLs on WorldPop...\n")
available_urls = []
for code in iso3_codes:
    url = f"https://data.worldpop.org/GIS/Population/Global_2000_2020/2020/{code}/"
    try:
        response = requests.head(url, timeout=5)
        if response.status_code == 200:
            print(f"[✅] FOUND: {url}")
            available_urls.append((code, url))
        else:
            print(f"[❌] NOT FOUND: {url} (Status {response.status_code})")
    except requests.RequestException as e:
        print(f"[ERROR] Could not check {url}: {e}")

# === Step 5: Final Summary
print(f"\n[RESULT] ✅ {len(available_urls)} valid WorldPop folders found:\n")
for code, url in available_urls:
    print(f"{code}: {url}")
