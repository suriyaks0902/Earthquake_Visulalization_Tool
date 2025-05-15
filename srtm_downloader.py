import os
import requests
import netrc
import boto3
from zipfile import ZipFile, BadZipFile


def upload_to_s3(local_path, bucket_name, s3_key):
    s3 = boto3.client('s3')
    try:
        s3.upload_file(local_path, bucket_name, s3_key)
        print(f"📤 Uploaded to S3: s3://{bucket_name}/{s3_key}")
    except Exception as e:
        print(f"❌ S3 upload failed for {local_path}: {e}")

def download_srtm_tile_nasa(tile_name, save_dir='data/srtm_nasa'):
    os.makedirs(save_dir, exist_ok=True)

    filename = f"{tile_name}.SRTMGL1.hgt.zip"
    url = f"https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{filename}"
    zip_path = os.path.join(save_dir, filename)
    hgt_path = os.path.join(save_dir, f"{tile_name}.hgt")

    if os.path.exists(hgt_path):
        print(f"✅ {tile_name} already exists.")
        return hgt_path

    print(f"🌐 Downloading {filename} from NASA...")

    try:
        # ✅ Parse credentials directly from .netrc
        creds = netrc.netrc().authenticators("urs.earthdata.nasa.gov")
        session = requests.Session()
        session.auth = (creds[0], creds[1])  # (username, password)

        # Download the .zip
        with session.get(url, stream=True, timeout=30) as response:
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            else:
                print(f"❌ Failed to download {tile_name}: {response.status_code}")
                return None

        # Extract the .hgt file
        try:
            with ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
        except BadZipFile:
            print("❌ Corrupt ZIP file.")
            return None

        # Confirm file was extracted
        if os.path.exists(hgt_path):
            print(f"✅ Extracted: {hgt_path}")

            # ✅ Upload to AWS S3
            bucket = "srtm-tiles-earthquake-vis"  
            s3_key = f"srtm_tiles/{os.path.basename(hgt_path)}"
            upload_to_s3(hgt_path, bucket, s3_key)

            return hgt_path
        else:
            print("❌ Extraction failed or file not found.")
            return None

    except Exception as e:
        print(f"❌ Error during download of {tile_name}: {e}")
        return None


def get_or_fetch_tile(tile_name, bucket, local_dir="data/srtm_nasa"):
    s3 = boto3.client("s3")
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, f"{tile_name}.hgt")

    # ✅ Check local first
    if os.path.exists(local_path):
        return local_path

    # ✅ Check S3 before downloading
    key = f"srtm_tiles/{tile_name}.hgt"
    try:
        s3.download_file(bucket, key, local_path)
        print(f"📥 Downloaded from S3: {key}")
        return local_path
    except Exception as e:
        print(f"🔍 Not in S3: {key} — {e}")

    # ❌ Not in S3, fetch from NASA
    path = download_srtm_tile_nasa(tile_name)
    if path:
        # ✅ Upload to S3 after successful download
        try:
            s3.upload_file(path, bucket, key)
            print(f"📤 Uploaded to S3: {key}")
        except Exception as e:
            print(f"⚠️ Failed to upload to S3: {e}")
        return path
    else:
        return None