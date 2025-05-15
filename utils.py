import boto3
from botocore.exceptions import ClientError
import pandas as pd
import numpy as np
import json
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def clean_dataframe_for_pydeck(df):
    df_clean = df.copy()

    # Drop rows with nulls in required columns
    df_clean = df_clean.dropna(subset=['latitude', 'longitude', 'depth', 'mag', 'place', 'type', 'time'])

    # Ensure all values are serializable primitive types
    df_clean['latitude'] = pd.to_numeric(df_clean['latitude'], errors='coerce').fillna(0).astype(float)
    df_clean['longitude'] = pd.to_numeric(df_clean['longitude'], errors='coerce').fillna(0).astype(float)
    df_clean['depth'] = pd.to_numeric(df_clean['depth'], errors='coerce').fillna(0).astype(float)
    df_clean['mag'] = pd.to_numeric(df_clean['mag'], errors='coerce').fillna(0).astype(float)

    # Convert to str (not object) — necessary for JSON serialization
    df_clean['place'] = df_clean['place'].astype(str)
    df_clean['type'] = df_clean['type'].astype(str)

    # Fix datetime
    df_clean['time'] = pd.to_datetime(df_clean['time'], errors='coerce')
    df_clean = df_clean[df_clean['time'].notnull()]
    df_clean['time'] = df_clean['time'].astype(str)

    return df_clean



def load_earthquake_data(filepath):

    df = pd.read_csv(filepath)
    df = df[['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'type']].copy()

    # Drop rows with missing key data
    df.dropna(subset=['time', 'latitude', 'longitude', 'depth', 'mag', 'place', 'type'], inplace=True)

    # Convert time to datetime and then to string
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df = df[df['time'].notnull()]
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    df['time'] = df['time'].astype(str)

    # Ensure all columns are basic types (float or str)
    df = df.astype({
        'latitude': 'float64',
        'longitude': 'float64',
        'depth': 'float64',
        'mag': 'float64',
        'place': 'str',
        'type': 'str'
    })

    return df


def plot_quake_histogram(df):
    df = pd.DataFrame(df)  # Ensure input is a DataFrame if passed from records
    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.date
    count_series = df.groupby('date').size()
    fig, ax = plt.subplots(figsize=(15, 5))
    count_series.plot(kind='bar', ax=ax, color="#1DB954")
    ax.set_title("Daily Earthquake Counts")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Earthquakes")
    fig.tight_layout()
    return fig


def plot_earthquake_map(df):
    df = pd.DataFrame(df)
    df['mag'] = pd.to_numeric(df['mag'], errors='coerce')

    # Filter out negative or zero magnitude (they can't be plotted as size)
    df = df[df['mag'] > 0]

    df['date'] = pd.to_datetime(df['time'], utc=True, errors='coerce').dt.date.astype(str)

    fig = px.scatter_geo(
        df,
        lat='latitude',
        lon='longitude',
        color='mag',
        size='mag',
        hover_name='place',
        animation_frame='date',
        projection='natural earth',
        title='Earthquake Events (Animated)'
    )
    return fig


def plot_3d_earthquake(df):
    df = pd.DataFrame(df)

    # Sanitize input: convert and filter numeric columns
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['depth'] = pd.to_numeric(df['depth'], errors='coerce')
    df['mag'] = pd.to_numeric(df['mag'], errors='coerce')

    # Drop rows with missing or invalid values
    df = df.dropna(subset=['longitude', 'latitude', 'depth', 'mag'])
    df = df[df['mag'] > 0]  # Only keep positive magnitudes
    df = df[df['depth'] >= 0]  # Optional: ignore negative depth (if it exists)

    fig = go.Figure(data=[go.Scatter3d(
        x=df['longitude'],
        y=df['latitude'],
        z=-df['depth'],  # Depth is shown inverted (deeper goes "down")
        mode='markers',
        marker=dict(
            size=df['mag'] * 2,
            color=df['mag'],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title='Magnitude')
        ),
        text=df['place']
    )])

    fig.update_layout(
        scene=dict(
            xaxis_title='Longitude',
            yaxis_title='Latitude',
            zaxis_title='Depth (km)'
        ),
        title='3D Earthquake Scatter Plot'
    )

    return fig

# Function to create a shockwave animation using Plotly
# This function creates a simple shockwave animation using Plotly
# based on a given latitude, longitude, and magnitude.
# It generates a series of frames that represent the shockwave expanding over time.




import numpy as np
from perlin_noise import PerlinNoise

def generate_synthetic_terrain(lat, lon, size_deg=1.0, resolution=100):
    x = np.linspace(lon - size_deg / 2, lon + size_deg / 2, resolution)
    y = np.linspace(lat - size_deg / 2, lat + size_deg / 2, resolution)
    xx, yy = np.meshgrid(x, y)

    noise = PerlinNoise(octaves=4)
    zz = np.array([
        [noise([i / resolution, j / resolution]) for j in range(resolution)]
        for i in range(resolution)
    ])
    zz *= 1000  # scale elevation

    return xx, yy, zz

import rasterio
import plotly.graph_objects as go
import numpy as np

def load_hgt_tile(hgt_path):
    with rasterio.open(hgt_path) as src:
        elevation = src.read(1)
        transform = src.transform

        # Generate coordinates
        nrows, ncols = elevation.shape
        xs = np.linspace(transform.c, transform.c + ncols * transform.a, ncols)
        ys = np.linspace(transform.f, transform.f + nrows * transform.e, nrows)
        xx, yy = np.meshgrid(xs, ys)

    return xx, yy, elevation

def plot_surface_preview(xx, yy, zz):
    fig = go.Figure(data=[go.Surface(z=zz, x=xx, y=yy, colorscale='Earth')])
    fig.update_layout(
        title="🗻 Surface Preview (SRTM 30m DEM)",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Elevation (m)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1))
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def tile_exists_in_s3(tile_name, bucket_name, prefix='srtm_tiles/'):
    s3 = boto3.client('s3')
    key = f"{prefix}{tile_name}.hgt"
    try:
        s3.head_object(Bucket=bucket_name, Key=key)
        return True
    except ClientError:
        return False