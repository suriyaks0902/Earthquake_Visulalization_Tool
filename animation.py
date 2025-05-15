import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import imageio
from shapely.geometry import Point
import contextily as ctx
from staticmap import StaticMap, CircleMarker
import os
import numpy as np
import plotly.graph_objects as go
from rasterio.transform import xy

def generate_animation(df, output='earthquake_animation.gif'):
    # Ensure time is in datetime format
    df['time'] = pd.to_datetime(df['time'], errors='coerce')

    # Drop any rows where time failed to parse
    df = df[df['time'].notnull()].copy()

    # Create date column
    df['date'] = df['time'].dt.date
    dates = sorted(df['date'].unique())
    images = []

    for date in dates:
        subset = df[df['date'] == date]
        plt.figure(figsize=(10, 5))
        plt.scatter(subset['longitude'], subset['latitude'], s=subset['mag']**2, alpha=0.6)
        plt.title(f"Earthquakes on {date}")
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.grid(True)
        plt.savefig('frame.png')
        images.append(imageio.imread('frame.png'))
        plt.close()

    imageio.mimsave(output, images, duration=0.5)


import numpy as np
import plotly.graph_objects as go
from rasterio.transform import xy
import rasterio
import os

def generate_grid_from_transform(transform, shape):
    height, width = shape
    xs = np.arange(width)
    ys = np.arange(height)
    xx, yy = np.meshgrid(xs, ys)

    lon, lat = xy(transform, yy, xx, offset='center')  # gets center of each cell
    lon_grid = np.array(lon).reshape(shape)
    lat_grid = np.array(lat).reshape(shape)
    return lon_grid, lat_grid

def plot_surface_preview(elevation, transform):
    xx, yy = generate_grid_from_transform(transform, elevation.shape)

    print("[DEBUG Preview] elevation.shape =", elevation.shape)
    print("[DEBUG Preview] xx/yy =", xx.shape, yy.shape)
    print("[DEBUG Preview] z min/max =", np.nanmin(elevation), np.nanmax(elevation))

    fig = go.Figure(data=[go.Surface(
        z=elevation,
        x=xx,
        y=yy,
        colorscale='Earth',
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.8),
        opacity=1.0
    )])

    fig.update_layout(
        title="🗻 Surface Preview (DEM Only)",
        scene=dict(
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            zaxis_title="Elevation (m)",
            camera=dict(eye=dict(x=1.6, y=1.6, z=0.9)),
            aspectmode="data"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig


def plot_dem_with_shockwave(elevation, transform, quake_lat, quake_lon, frames=25):
    # 🌍 Generate geographic grid
    xx, yy = generate_grid_from_transform(transform, elevation.shape)

    # ✅ Debug
    print("[DEBUG] elevation:", elevation.shape, "xx:", xx.shape, "yy:", yy.shape)
    print("[DEBUG] elevation min:", np.min(elevation), "max:", np.max(elevation))

    # 🎛️ Surface
    surface = go.Surface(
        z=elevation,
        x=xx,
        y=yy,
        colorscale='Earth',
        opacity=1.0,
        showscale=False
    )

    # 🔁 Ripple animation
    ripple_frames = []
    max_radius = 0.5  # degrees
    z_top = np.nanmax(elevation) + 100  # hover ripple above terrain

    for f in range(frames):
        radius = (f + 1) / frames * max_radius
        angle = np.linspace(0, 2 * np.pi, 200)
        ripple_x = quake_lon + radius * np.cos(angle)
        ripple_y = quake_lat + radius * np.sin(angle)
        ripple_z = np.full_like(ripple_x, z_top)

        ripple = go.Scatter3d(
            x=ripple_x,
            y=ripple_y,
            z=ripple_z,
            mode='lines',
            line=dict(color='red', width=5),
            name=f"Ripple {f + 1}"
        )
        ripple_frames.append(go.Frame(data=[ripple]))

    fig = go.Figure(
        data=[surface],
        layout=go.Layout(
            title="🌋 Earthquake Shockwave over Real Terrain",
            scene=dict(
                xaxis_title='Longitude',
                yaxis_title='Latitude',
                zaxis_title='Elevation (m)',
                camera=dict(eye=dict(x=1.3, y=1.3, z=0.8)),
                aspectmode="data"
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            updatemenus=[dict(
                type='buttons',
                buttons=[{
                    'label': '▶️ Play',
                    'method': 'animate',
                    'args': [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                }],
                showactive=False
            )]
        ),
        frames=ripple_frames
    )

    return fig