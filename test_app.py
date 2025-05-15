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
import plotly.express as px
from utils import load_earthquake_data
from animation import generate_animation, plot_surface_preview, plot_dem_with_shockwave
from terrain_loader import get_tile_name
# from srtm_downloader import download_srtm_tile_nasa, get_or_fetch_tile
from scipy.optimize import fsolve

from geopy.geocoders import Nominatim
import rasterio
from shapely.geometry import Point, mapping
import geopandas as gpd
from rasterio.mask import mask
import reverse_geocoder as rg
import concurrent.futures
from gmpe_radius_estimator import estimate_radius_pga_gmpe
import plotly.graph_objects as go


st.set_page_config(
    page_title="Global Earthquake Visualization Dashboard",
    layout="wide",
    page_icon="🌍"
)

# Constants
BUCKET_NAME = "srtm-tiles-earthquake-vis"
MAPBOX_TOKEN = "pk.eyJ1Ijoic3VyaXlha3MwOTAyIiwiYSI6ImNtOTJhNXN2NjAzc2kycm9sOW9ya2ZjOTYifQ.StDVD37oxbNEnoiGWDChSA"
TILE_CACHE_FILE = "cached_s3_tiles.json"
POP_ESTIMATE_CACHE = "cached_population_estimates.csv"

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

def classify_earthquake(mag):
    if mag < 2.0:
        return "Micro"
    elif mag < 4.0:
        return "Minor"
    elif mag < 5.0:
        return "Light"
    elif mag < 6.0:
        return "Moderate"
    elif mag < 7.0:
        return "Strong"
    elif mag < 8.0:
        return "Major"
    elif mag < 9.0:
        return "Great"
    else:
        return "Massive"



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

# def estimate_radius_from_mmi(mag, target_mmi=5.0):
#     def mmi_equation(R):
#         R = max(R, 1e-3)  # avoid log(0)
#         return 3.66 + 1.66 * mag - 1.3 * np.log10(R) - 0.00255 * R - target_mmi
#     R0 = fsolve(mmi_equation, x0=10)[0]
#     return round(R0, 2)

def estimate_radius_from_mmi(mag, lat, lon):
    return estimate_radius_pga_gmpe(mag, lat, lon, target_pga=0.05)

geolocator = Nominatim(user_agent="eq_visualizer")

# def get_country_code(lat, lon):
#     location = geolocator.reverse((lat, lon), language="en", timeout=10)
#     return location.raw['address']['country_code'].upper() if location and 'country_code' in location.raw['address'] else None

def fast_get_country(lat, lon):
    return rg.search((lat, lon), mode=1)[0]['cc'].upper()


def render_mapbox_terrain_html(lat, lon, mag, token, show_shockwave, show_rings, spread_km=50):
    with open("mapbox_3d_terrain.html", "r", encoding="utf-8") as f:
        template = Template(f.read())
    html = template.render(LAT=lat, LON=lon, MAG=mag, TOKEN=token, SHOW_SHOCKWAVE=show_shockwave, SHOW_RINGS=show_rings, SPREAD_KM=spread_km)
    return html

# --- PAGE NAVIGATION ---
st.sidebar.title("Earthquake Dashboard Navigation")
page = st.sidebar.radio("Select Page", ["🌍 Overview", "👥 Population Impact", "📌 Country Insights"])

# --- FILE UPLOAD + BASE DATA ---
st.sidebar.title("Upload & Filters")
uploaded_file = st.sidebar.file_uploader("Upload Earthquake CSV", type=["csv"])

if uploaded_file:
    df = load_earthquake_data(uploaded_file)
    df = clean_dataframe_for_pydeck(df)
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.date

    if "with_tiles" in uploaded_file.name:
        df_with_tiles = df.copy()
    else:
        s3_tile_set = load_tile_set_from_s3(BUCKET_NAME)
        df_with_tiles = df[df['place'].isin(filter_earthquakes_with_available_tiles(df, s3_tile_set, limit=2000))].copy()

    # UI and logic: Use parallel fast estimation if cache does not exist
    if os.path.exists(POP_ESTIMATE_CACHE):
        cached_df = pd.read_csv(POP_ESTIMATE_CACHE)
        print(cached_df['estimated_population'].isna().sum(), "/", len(cached_df))
        df_with_tiles = pd.merge(
            df_with_tiles,
            cached_df[['time', 'latitude', 'longitude', 'estimated_population', 'country_code']],
            on=['time', 'latitude', 'longitude'],
            how='left'
        )
    # else:
    #     st.markdown(" Estimating population impact (within 10km) using parallel fast estimation...")
    #     estimated_df = estimate_population_parallel(df_with_tiles.copy())
    #     df_with_tiles = pd.merge(
    #         df_with_tiles,
    #         estimated_df[['time', 'latitude', 'longitude', 'estimated_population', 'country_code']],
    #         on=['time', 'latitude', 'longitude'],
    #         how='left'
    #     )
    #     estimated_df[['time', 'latitude', 'longitude', 'estimated_population', 'country_code']].to_csv(POP_ESTIMATE_CACHE, index=False)


    # Safely find max population row
    valid_pop_df = df_with_tiles.dropna(subset=['estimated_population'])

    if not valid_pop_df.empty:
        max_index = valid_pop_df['estimated_population'].idxmax()
        most_affected_region = valid_pop_df.loc[max_index,'country_code']
    else:
        most_affected_region = "N/A"


    # Apply magnitude and depth filtering first
    magnitude_range = st.sidebar.slider("Magnitude Range", 0.0, 10.0, (0.0, 10.0), 0.1)
    depth_range = st.sidebar.slider("Depth Range (km)", 0, 700, (0, 700))

    range_filtered_df = df_with_tiles[
        (df_with_tiles['mag'] >= magnitude_range[0]) &
        (df_with_tiles['mag'] <= magnitude_range[1]) &
        (df_with_tiles['depth'] >= depth_range[0]) &
        (df_with_tiles['depth'] <= depth_range[1])
    ]

    # Dropdown should show places within the filtered range
    available_places = sorted(range_filtered_df['place'].unique())
    selected_place = st.sidebar.selectbox("Place (only DEM-available)", options=["All"] + available_places)

    # Now filter by place too (if not "All")
    if selected_place != "All":
        filtered_df = range_filtered_df[range_filtered_df['place'] == selected_place]
    else:
        filtered_df = range_filtered_df


    if page == "🌍 Overview":
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("#### 🌐 Global Earthquake Map")
            if not filtered_df.empty:
                selected_eq = filtered_df.iloc[0]
                lat, lon, mag = selected_eq['latitude'], selected_eq['longitude'], selected_eq['mag']
                show_shockwave = st.toggle("Show Shockwave Effect", value=False, key="toggle_shockwave")
                show_rings = st.toggle("Show Concentric Rings", value=False, key="toggle_rings")

                spread_km = estimate_radius_from_mmi(mag, lat, lon)
                html = render_mapbox_terrain_html(
                    lat, lon, mag, MAPBOX_TOKEN,
                    show_shockwave=show_shockwave,
                    show_rings=show_rings,
                    spread_km=spread_km
                )

                # st.components.v1.html(html, height=600)
                timestamp = int(time.time())
                st.components.v1.html(html + f"\n<!-- cache-bust-{timestamp} -->", height=600)
            else:
                st.warning("No data available for the current filters.")

            st.subheader("🌐 High-Quality 2D Animated Earthquake Map")
            # df_2d_layer = filtered_df[['longitude', 'latitude', 'mag', 'place']].copy()
            df_2d_layer = df_with_tiles[['longitude', 'latitude', 'mag', 'place']].copy()

            view_lat = df_2d_layer['latitude'].mean()
            view_lon = df_2d_layer['longitude'].mean()

            if selected_place != "All" and not filtered_df.empty:
                eq = filtered_df[filtered_df['place'] == selected_place].iloc[0]
                view_lat = eq['latitude']
                view_lon = eq['longitude']

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
                zoom=2,
                pitch=30,
            )
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "Magnitude: {mag}\nLocation: {place}"}
            )
            st.pydeck_chart(r)

        with col2:
            st.metric("Total Earthquakes", len(filtered_df))
            most_place = filtered_df['place'].value_counts().idxmax() if not filtered_df.empty else "N/A"
            st.metric("Most Affected", most_place)
            st.metric("Average Magnitude", f"{filtered_df['mag'].mean():.2f}" if not filtered_df.empty else "N/A")
            st.metric("Max Depth", f"{filtered_df['depth'].max():.1f} km" if not filtered_df.empty else "N/A")
            st.metric("Estimated Shockwave Radius", f"{spread_km:.1f} km")
            st.metric("Earthquake Class", classify_earthquake(selected_eq["mag"]))
            st.subheader("Earthquake Magnitude Classification chart")

            mag_classes = {
                "Micro": (0.0, 2.0),
                "Minor": (2.0, 4.0),
                "Light": (4.0, 5.0),
                "Moderate": (5.0, 6.0),
                "Strong": (6.0, 7.0),
                "Major": (7.0, 8.0),
                "Great": (8.0, 9.0),
                "Massive": (9.0, 10.0)
            }

            colors = ["#D0D0D0", "#A0A0FF", "#60C0FF", "#00BFFF", "#FFA500", "#FF4500", "#FF0000", "#800000"]

            fig = go.Figure()
            for i, (label, (start, end)) in enumerate(mag_classes.items()):
                fig.add_trace(go.Bar(
                    x=[end - start],
                    y=[label],
                    orientation='h',
                    marker=dict(color=colors[i]),
                    hovertemplate=f"{label}: {start} - {end} M<extra></extra>",
                    name=label
                ))

            fig.update_layout(
                barmode='stack',
                height=200,
                showlegend=False,
                xaxis=dict(title="Magnitude Range", range=[0, 10]),
                yaxis=dict(title="", showgrid=False),
                margin=dict(l=20, r=20, t=20, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)


    elif page == "👥 Population Impact":
        # st.markdown("### 👥 Estimated Population Impact View")
        exec(open("page_2_population.py", encoding="utf-8").read())


    elif page == "📌 Country Insights":
        exec(open("page_3_country_insights.py", encoding="utf-8").read())

else:
    st.info("Upload a valid USGS earthquake CSV file to explore the dashboard.")
