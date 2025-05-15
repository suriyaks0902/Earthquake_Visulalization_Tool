import streamlit as st
import time
import numpy as np
import base64
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import pydeck as pdk
from jinja2 import Template
import json
import os
import boto3
from utils import load_earthquake_data, plot_quake_histogram, tile_exists_in_s3
from animation import generate_animation, plot_surface_preview, plot_dem_with_shockwave
from terrain_loader import get_tile_name, load_local_terrain
from srtm_downloader import download_srtm_tile_nasa, get_or_fetch_tile
from scipy.optimize import fsolve

st.set_page_config(
    page_title="Earthquake Visualization Dashboard",
    layout="wide",
    page_icon="🌍"
)

# Constants
BUCKET_NAME = "srtm-tiles-earthquake-vis"  # 🔁 Replace with your actual bucket name
MAPBOX_TOKEN = "pk.eyJ1Ijoic3VyaXlha3MwOTAyIiwiYSI6ImNtOTJhNXN2NjAzc2kycm9sOW9ya2ZjOTYifQ.StDVD37oxbNEnoiGWDChSA"
TILE_CACHE_FILE = "cached_s3_tiles.json"

# Utility Functions
@st.cache_data(show_spinner=False)

def load_tile_set_from_s3(bucket):
    if os.path.exists(TILE_CACHE_FILE):
        with open(TILE_CACHE_FILE, 'r') as f:
            return set(json.load(f))
    s3 = boto3.client("s3")
    result = s3.list_objects_v2(Bucket=bucket, Prefix="srtm_tiles/")
    tile_names = [obj['Key'].split('/')[-1].replace(".hgt", "").upper() for obj in result.get('Contents', [])]
    with open(TILE_CACHE_FILE, 'w') as f:
        json.dump(list(tile_names), f)
    return set(tile_names)

def filter_earthquakes_with_available_tiles(df, tile_set, limit=None):
    available_places = []
    subset = df if limit is None else df.head(limit)
    for _, row in subset.iterrows():
        tile = get_tile_name(row['latitude'], row['longitude'])
        if tile in tile_set:
            available_places.append(row['place'])
    return sorted(set(available_places))

def clean_dataframe_for_pydeck(df):
    df_clean = df.copy()
    df_clean = df_clean.astype({
        'latitude': float,
        'longitude': float,
        'depth': float,
        'mag': float,
        'place': str,
        'type': str
    })
    df_clean = df_clean[df_clean['time'].notnull()]
    df_clean['time'] = pd.to_datetime(df_clean['time'], errors='coerce')
    df_clean = df_clean[df_clean['time'].notnull()]
    df_clean['time'] = df_clean['time'].astype(str)
    return df_clean

def create_shockwave_layer(lat, lon, radius, opacity):
    red_intensity = int(255 * opacity)
    return pdk.Layer(
        "ScatterplotLayer",
        data=pd.DataFrame([{ "latitude": lat, "longitude": lon, "radius": radius }]),
        get_position='[longitude, latitude]',
        get_radius='radius',
        get_fill_color=f'[255, {255 - red_intensity}, 0, {int(255 * opacity)}]',
        pickable=False,
        opacity=opacity,
        stroked=True,
        filled=False,
        line_width_min_pixels=2,
    )
def estimate_radius_from_mmi(mag, target_mmi=5.0):
    def mmi_equation(R):
        return 3.66 + 1.66 * mag - 1.3 * np.log10(R) - 0.00255 * R - target_mmi
    R0 = fsolve(mmi_equation, x0=10)[0]
    return round(R0, 2)

def render_mapbox_terrain_html(lat, lon , mag, token, show_shockwave=True, spread_km=50):
    with open("mapbox_3d_terrain.html", "r") as f:
        template = Template(f.read())
    html = template.render(LAT=lat, LON=lon, MAG=mag, TOKEN=token, SHOW_SHOCKWAVE=str(show_shockwave).lower(), SPREAD_KM=spread_km)
    return html

def show_looping_shockwave(df, selected_place):
    quake = df[df['place'] == selected_place]
    if quake.empty:
        st.error("Selected earthquake not found.")
        return

    quake = quake.iloc[0]
    lat, lon, mag = quake['latitude'], quake['longitude'], quake['mag']

    st.markdown(f"### Live Looping Shockwave: {selected_place} (Magnitude {mag})")

    max_radius = mag * 150000
    frames = 30
    chart_placeholder = st.empty()

    ripple_opacity_decay = 0.07
    ripple_spacing = int(frames / 5)

    while True:
        layers = []
        for i in range(frames):
            radius = (i + 1) / ripple_opacity_decay * max_radius
            opacity = max(0.1, 1.0 - ripple_opacity_decay * i)
            ripple_layer = create_shockwave_layer(lat, lon, radius, opacity)
            layers.append(ripple_layer)

            view_state = pdk.ViewState(
                latitude=lat,
                longitude=lon,
                zoom=6,
                pitch=60,
                bearing=45
            )

            deck = pdk.Deck(
                layers=layers[-5:],
                initial_view_state=view_state,
                map_style="mapbox://styles/mapbox/satellite-v9",
            )

            chart_placeholder.pydeck_chart(deck)
            time.sleep(0.15)

            if st.session_state.get("stop_loop", False):
                return

# UI
st.markdown("""
    <h1 style='text-align: center; color: #1DB954;'>🌍 Earthquake Visualization Dashboard</h1>
    <h4 style='text-align: center; color: #cccccc;'>Visualizing global seismic activity through space and time</h4>
""", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.file_uploader(
    "### 📂 Upload your USGS Earthquake CSV file:",
    type=["csv"],
    help="Download from https://earthquake.usgs.gov/earthquakes/feed/v1.0/csv.php"
)

if uploaded_file:
    df = load_earthquake_data(uploaded_file)
    df = clean_dataframe_for_pydeck(df)
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.date

    st.success(f"📁 Earthquake dataset loaded: {len(df)} rows")
    st.markdown("🔍 **Checking SRTM availability in your S3 bucket...**")

    # ⚡ Filter earthquakes with available SRTM tiles (limit scanning to 500 for speed)
    s3_tile_set = load_tile_set_from_s3(BUCKET_NAME)
    valid_places = filter_earthquakes_with_available_tiles(df, s3_tile_set, limit=500)

    # 📌 Filter the full df using valid places
    df_with_tiles = df[df['place'].isin(valid_places)].copy()
    st.success(f"✅ Found {len(df_with_tiles)} earthquakes with matching DEM tiles out of {len(df)} total")

    # 💾 Save filtered dataset locally
    df_with_tiles.to_csv("earthquakes_with_tiles.csv", index=False)
    st.success("✅ Filtered dataset saved as `earthquakes_with_tiles.csv`")
    # 📊 Display summary
    st.markdown(f"""
    ### 📊 Tile Match Summary  
    - Total Earthquakes: `{len(df)}`  
    - DEM-Available Earthquakes: `{len(df_with_tiles)}`  
    - Coverage: `{len(df_with_tiles)/len(df)*100:.2f}%`  
    """)

    with st.expander("📄 Preview Earthquakes with DEM Tiles"):
        st.dataframe(df_with_tiles[['time', 'latitude', 'longitude', 'mag', 'place']], height=300)


    with st.expander("🔍 Preview Raw Earthquake Data"):
        st.dataframe(df, height=300)

    st.markdown("---")
    st.subheader("🌐 High-Quality 2D Animated Earthquake Map")

    df_2d_layer = df[['longitude', 'latitude', 'mag', 'place']].copy()
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df_2d_layer.to_dict("records"),
        get_position='[longitude, latitude]',
        get_color='[255, 140, 0, 140]',
        get_radius='mag * 10000',
        pickable=True,
        opacity=0.6,
    )

    view_state = pdk.ViewState(
        latitude=df['latitude'].mean(),
        longitude=df['longitude'].mean(),
        zoom=1.5,
        pitch=30,
    )
    r = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip={"text": "Magnitude: {mag}\nLocation: {place}"}
    )
    st.pydeck_chart(r)

    st.markdown("---")
    st.subheader(" Earthquake Shockwave Animation (Real Terrain)")

    if MAPBOX_TOKEN is None:
        st.warning("MAPBOX_PUBLIC_TOKEN environment variable not set.")
    else:
        selected_place = st.selectbox("Select Earthquake Location (SRTM Available):", valid_places)

        # show_shockwave = st.checkbox("Show Earthquake Shockwave", value=True)
        show_shockwave = st.toggle("Show Shockwave Effect", value=True, key="toggle_shockwave")
        target_mmi = st.slider("Minimum MMI to Visualize", 3.0, 8.0, 5.0, 0.5)
        opacity = st.slider("Shockwave Opacity", 0.0, 1.0, 0.5, 0.1)


        quake = df_with_tiles[df_with_tiles['place'] == selected_place].iloc[0]
        lat, lon, mag = quake["latitude"], quake["longitude"], quake["mag"]
        spread_km = estimate_radius_from_mmi(mag, target_mmi)
        html_code = render_mapbox_terrain_html(lat, lon, mag, MAPBOX_TOKEN, show_shockwave, spread_km)
        st.components.v1.html(html_code, height=600)
        st.markdown(f"📍 Epicenter: **{selected_place}** (Lat: {lat}, Lon: {lon})")


    st.markdown("---")
    st.subheader("🎥 Export Animated Earthquake GIF")
    if st.button("Generate Animation"):
        generate_animation(df)
        st.success("GIF created: earthquake_animation.gif")
        with open("earthquake_animation.gif", "rb") as f:
            st.download_button(
                label="🔗 Download Earthquake Animation GIF",
                data=f,
                file_name="earthquake_animation.gif",
                mime="image/gif"
            )
else:
    st.info("Please upload a CSV file from the USGS Earthquake feed to continue.")


    # st.markdown("---")

    # st.subheader("📊 Earthquake Frequency Histogram")
    # fig3 = plot_quake_histogram(df)
    # st.pyplot(fig3)

    # st.markdown("---")

    # st.subheader("🗺 Plotly Earthquake Map with Time Animation")
    # fig_map = plot_earthquake_map(df)
    # st.plotly_chart(fig_map, use_container_width=True)

    # st.markdown("---")

    # st.subheader("🛰 Interactive 3D Earthquake Scatter Plot")
    # fig_3d = plot_3d_earthquake(df)
    # st.plotly_chart(fig_3d, use_container_width=True)


