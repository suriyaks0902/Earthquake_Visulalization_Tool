
# import streamlit as st
# import pandas as pd
# import numpy as np
# import pydeck as pdk
# import plotly.express as px
# from gmpe_radius_estimator import estimate_radius_pga_gmpe

# @st.cache_data
# def load_data():
#     df = pd.read_csv("earthquakes_with_population.csv")
#     numeric_cols = ["estimated_population", "mag", "depth", "fault_distance_km", "plate_distance_km", "log_population"]
#     df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
#     df["display"] = df["place"] + " | " + df["time"]
#     return df

# df = load_data()

# # st.set_page_config(page_title="Population Impact", layout="wide")
# st.title("Population Impact of Earthquakes")

# selected_eq = st.selectbox("Select Earthquake", df["display"])
# eq = df[df["display"] == selected_eq].iloc[0]

# col1, col2, col3 = st.columns([2.5, 1.5, 1.5])

# with col1:
#     st.subheader("Earthquake Locations Map")
#     quake_map = pdk.Deck(
#         map_style="mapbox://styles/mapbox/dark-v10",
#         initial_view_state=pdk.ViewState(
#             latitude=eq["latitude"],
#             longitude=eq["longitude"],
#             zoom=2.5,
#             pitch=30,
#         ),
#         layers=[
#             pdk.Layer(
#                 "ScatterplotLayer",
#                 data=df,
#                 get_position='[longitude, latitude]',
#                 get_color='[255, 100, 100, 160]',
#                 get_radius=50000,
#             ),
#         ],
#     )
#     st.pydeck_chart(quake_map)

#     st.subheader("Top 5 Earthquakes by Magnitude")
#     st.dataframe(df.nlargest(5, "mag")[["place", "mag"]].rename(columns={"place": "Location", "mag": "Magnitude"}))

#     st.subheader("Top 5 by Population Affected")
#     st.dataframe(df.nlargest(5, "estimated_population")[["place", "estimated_population"]].rename(
#         columns={"place": "Location", "estimated_population": "Population Affected"}))

# with col2:
#     st.subheader("Key Metrics")
#     st.metric("Total Earthquakes", len(df))
#     st.metric("Strongest Earthquake", f"{df['mag'].max():.1f} M")
#     st.metric("Most Affected Region", df.loc[df['estimated_population'].idxmax(), 'place'])

#     st.subheader("Heatmap: Fault Distance vs Earthquake Magnitude")
#     st.caption("This heatmap shows how earthquake strength varies with proximity to fault lines. Most strong quakes occur closer to faults.")
#     # fig1 = px.density_heatmap(df.dropna(subset=["fault_distance_km", "mag"]),
#     #                           x="fault_distance_km", y="mag",
#     #                           nbinsx=30, nbinsy=20, color_continuous_scale="Inferno")
#     fig1 = px.density_heatmap(
#         df.dropna(subset=["fault_distance_km", "mag"]),
#         x="fault_distance_km", y="mag",
#         nbinsx=30, nbinsy=20,
#         color_continuous_scale="YlOrRd",
#         labels={
#             "fault_distance_km": "Distance from Fault (km)",
#             "mag": "Earthquake Magnitude"
#         }
#     )
#     st.plotly_chart(fig1, use_container_width=True)

#     st.subheader("Depth vs Tectonic Plate Distance")
#     st.caption("Deeper earthquakes usually occur near plate boundaries where subduction happens. Distant, shallow ones may be intraplate events.")
#     # fig2 = px.scatter(df, x="plate_distance_km", y="depth", color="mag", size="estimated_population",
#     #                   labels={"plate_distance_km": "Distance from Plate (km)", "depth": "Depth (km)"},
#     #                   color_continuous_scale="Turbo")
#     # st.plotly_chart(fig2, use_container_width=True)
#     # fig2 = px.scatter(
#     #     df.dropna(subset=["plate_distance_km", "depth", "mag", "estimated_population"]),
#     #     x="plate_distance_km", y="depth",
#     #     color="mag", size="estimated_population",
#     #     labels={"plate_distance_km": "Distance from Plate (km)", "depth": "Depth (km)"},
#     #     color_continuous_scale="Turbo")
#     fig2_df = df.dropna(subset=["plate_distance_km", "depth", "estimated_population", "mag"])
#     fig2_df = fig2_df[fig2_df["estimated_population"] > 0]
#     fig2_df["size_pop"] = fig2_df["estimated_population"].clip(lower=1)

#     fig2 = px.scatter(
#         fig2_df,
#         x="plate_distance_km", y="depth",
#         color="mag", size="size_pop",
#         labels={"plate_distance_km": "Distance from Plate (km)", "depth": "Depth (km)"},
#         color_continuous_scale="Turbo"
#     )
#     st.plotly_chart(fig2, use_container_width=True)

# with col3:
#     st.subheader("Population vs Fault Proximity")
#     df_valid3 = df[(df["mag"] > 0) & df["fault_distance_km"].notna() & df["estimated_population"].notna()]
#     # fig3 = px.scatter(
#     #     df_valid3,
#     #     x="fault_distance_km", y="estimated_population",
#     #     color="mag", size="mag",
#     #     labels={"fault_distance_km": "Distance from Fault (km)", "estimated_population": "Population Affected"},
#     #     color_continuous_scale="YlOrRd")
#     fig3_df = df.dropna(subset=["fault_distance_km", "estimated_population", "mag"])
#     fig3_df = fig3_df[(fig3_df["mag"] > 0) & (fig3_df["estimated_population"] > 0)]
#     fig3_df["size_mag"] = fig3_df["mag"].clip(lower=0.1)

#     fig3 = px.scatter(
#         fig3_df,
#         x="fault_distance_km", y="estimated_population",
#         color="mag", size="size_mag",
#         labels={"fault_distance_km": "Distance from Fault (km)", "estimated_population": "Population Affected"},
#         color_continuous_scale="YlOrRd"
#     )
#     st.plotly_chart(fig3, use_container_width=True)

#     st.subheader("Depth vs Fault Distance Line")
#     # Clean + sort data
#     line_df = df.dropna(subset=["fault_distance_km", "depth"]).sort_values("fault_distance_km")
#     fig4 = px.line(
#         df.dropna(subset=["fault_distance_km", "depth"]).sort_values("fault_distance_km"),
#         x="fault_distance_km", y="depth",
#         labels={
#             "fault_distance_km": "Distance from Fault (km)",
#             "depth": "Depth (km)"
#         },
#         line_shape="linear",  # or remove this line
#     )
#     st.plotly_chart(fig4, use_container_width=True)



#     st.subheader("Log-Transformed Population vs Fault Proximity")
#     # df_valid_log = df[(df["mag"] > 0) & df["fault_distance_km"].notna() & df["log_population"].notna()]
#     log_df = df[(df["log_population"].notnull()) & (df["mag"] > 0)]
#     log_df["size_mag"] = log_df["mag"].clip(lower=0.1)

#     # fig_log = px.scatter(
#     #     df_valid_log,
#     #     x="fault_distance_km", y="log_population",
#     #     color="mag", size="mag",
#     #     labels={"fault_distance_km": "Distance from Fault (km)", "log_population": "Log(Population + 1)"},
#     #     color_continuous_scale="Plasma")
#     fig_log = px.scatter(
#         log_df,
#         x="fault_distance_km", y="log_population",
#         color="log_population", color_continuous_scale="Purples",
#         size="size_mag",
#         labels={
#             "fault_distance_km": "Distance from Fault (km)",
#             "log_population": "Log(Population + 1)"
#         }
#     )
#     st.plotly_chart(fig_log, use_container_width=True)



import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import geopandas as gpd
import json
import os

@st.cache_data
def load_data():
    df = pd.read_csv("earthquakes_with_population.csv")
    numeric_cols = ["estimated_population", "mag", "depth", "fault_distance_km", "plate_distance_km", "log_population"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["display"] = df["place"] + " | " + df["time"]
    return df

def prepare_and_cache_geojson(shp_path, json_path, simplify_tolerance=0.01):
    """
    Load a shapefile, simplify the geometry, and cache as GeoJSON.
    """
    if not os.path.exists(json_path):
        gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
        gdf["geometry"] = gdf["geometry"].simplify(simplify_tolerance)
        geojson = json.loads(gdf.to_json())
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            geojson = json.load(f)
    return geojson

# === Create cache directory if it doesn't exist ===
os.makedirs("cache", exist_ok=True)

# === Define input shapefiles and output json paths ===
fault_shapefile = "fault_lines/Qfaults_US_Database.shp"
plate_shapefile = "plate_boundaries/PB2002_boundaries.shp"
fault_json = "cache/fault_lines.json"
plate_json = "cache/plate_boundaries.json"

# === Process and cache both ===
fault_geojson = prepare_and_cache_geojson(fault_shapefile, fault_json)
plate_geojson = prepare_and_cache_geojson(plate_shapefile, plate_json)

df = load_data()
st.title("Population Impact and Geophysical Insights")

# === Row 1 ===
col1, col2 = st.columns([2, 1])
with col1:
    st.subheader("🌍 Earthquake Impact Density Map")
    # === Load fault lines and plate boundaries ===
    faults_gdf = gpd.read_file("fault_lines/Qfaults_US_Database.shp").to_crs("EPSG:4326")
    plates_gdf = gpd.read_file("plate_boundaries/PB2002_boundaries.shp").to_crs("EPSG:4326")

    # Simplify GeoDataFrames (if using shapefiles)
    faults_gdf["geometry"] = faults_gdf["geometry"].simplify(0.01)
    plates_gdf["geometry"] = plates_gdf["geometry"].simplify(0.01)

    # === Sample earthquake data to reduce message size ===
    heat_df = df.dropna(subset=["latitude", "longitude", "estimated_population"])
    heat_df = heat_df.sample(min(5000, len(heat_df)), random_state=42)


    # === Create Heatmap Layer ===
    heat_layer = pdk.Layer(
        "HeatmapLayer",
        data=heat_df,
        get_position='[longitude, latitude]',
        get_weight='estimated_population',
        radiusPixels=60,
        threshold=0.05
    )

    # === Convert shapefiles to GeoJSON pydeck-compatible layers ===
    faults_layer = pdk.Layer(
        "GeoJsonLayer",
        data=fault_geojson,
        get_line_color=[255, 0, 0, 255],
        get_line_width=15,
        pickable=True,
        auto_highlight=True
    )

    plates_layer = pdk.Layer(
        "GeoJsonLayer",
        data=plate_geojson,
        get_line_color=[0, 255, 255, 255],
        get_line_width=15,
        pickable=True,
        auto_highlight=True
    )

    # === Set View State ===
    view_state = pdk.ViewState(
        latitude=heat_df["latitude"].mean(),
        longitude=heat_df["longitude"].mean(),
        zoom=3.5,
        pitch=30
    )

    # === Combine and Render Deck Map ===
    deck = pdk.Deck(
        layers=[heat_layer, faults_layer, plates_layer],
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10'
    )

    st.pydeck_chart(deck)

with col2:
    st.subheader("Avg. Magnitude by Fault Distance Bin")
    df_valid = df.dropna(subset=["fault_distance_km", "mag"])
    df_valid["fault_bin"] = pd.cut(df_valid["fault_distance_km"], bins=[0, 100, 500, 1000, 2000, 5000, 10000], right=False)
    hist = df_valid.groupby("fault_bin", observed=True)["mag"].mean().reset_index()
    hist["fault_bin"] = hist["fault_bin"].astype(str)
    fig_hist = px.bar(hist, x="fault_bin", y="mag",
                      labels={"mag": "Avg. Magnitude", "fault_bin": "Fault Distance (km)"},
                      color="mag", color_continuous_scale="OrRd")
    fig_hist.update_layout(
        coloraxis_colorbar=dict(title="Avg Magnitude"),
        xaxis_title="Distance from Fault (km)",
        yaxis_title="Average Earthquake Magnitude",
        legend_title_text="Magnitude Buckets"
    )
    st.plotly_chart(fig_hist, use_container_width=True)

# === Row 2 ===
col3, col4 = st.columns(2)
with col3:
    st.subheader("📉 Trend: Depth vs Fault Distance")
    trend_df = df.dropna(subset=["depth", "fault_distance_km"])
    lowess = sm.nonparametric.lowess(trend_df["depth"], trend_df["fault_distance_km"], frac=0.25)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=trend_df["fault_distance_km"], y=trend_df["depth"], mode="markers", opacity=0.3, name="Raw"))
    fig1.add_trace(go.Scatter(x=lowess[:, 0], y=lowess[:, 1], mode="lines", name="LOWESS", line=dict(color="cyan")))
    fig1.update_layout(xaxis_title="Distance from Fault (km)", yaxis_title="Depth (km)")
    
    st.plotly_chart(fig1, use_container_width=True)

with col4:
    st.subheader("📈 Trend: Magnitude vs Plate Distance")
    plate_df = df.dropna(subset=["plate_distance_km", "mag"])
    lowess_mag = sm.nonparametric.lowess(plate_df["mag"], plate_df["plate_distance_km"], frac=0.3)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=plate_df["plate_distance_km"], y=plate_df["mag"], mode="markers", opacity=0.3, name="Raw"))
    fig2.add_trace(go.Scatter(x=lowess_mag[:, 0], y=lowess_mag[:, 1], mode="lines", name="LOWESS", line=dict(color="orange")))
    fig2.update_layout(xaxis_title="Distance from Plate (km)", yaxis_title="Magnitude")
    st.plotly_chart(fig2, use_container_width=True)

# === Row 3 Full ===
st.subheader("👥 Trend: Population Affected vs Fault Distance")
pop_df = df.dropna(subset=["estimated_population", "fault_distance_km"])
# lowess_pop = sm.nonparametric.lowess(pop_df["estimated_population"], pop_df["fault_distance_km"], frac=0.3)

threshold = pop_df["estimated_population"].quantile(0.95)
pop_df_vis = pop_df[pop_df["estimated_population"] <= threshold]
lowess_pop = sm.nonparametric.lowess(pop_df_vis["estimated_population"], pop_df_vis["fault_distance_km"], frac=0.3)


tab1, tab2 = st.tabs(["🔍 Log Scale View", "📈 Raw Scale View"])
with tab1:
    fig_log = go.Figure()
    fig_log.add_trace(go.Scatter(x=pop_df["fault_distance_km"], y=pop_df["estimated_population"], mode="markers", opacity=0.3, name="Raw"))
    fig_log.add_trace(go.Scatter(x=lowess_pop[:, 0], y=lowess_pop[:, 1], mode="lines", name="LOWESS", line=dict(color="purple")))
    fig_log.update_layout(yaxis_type="log", xaxis_title="Distance from Fault (km)", yaxis_title="Estimated Population (log)")
    st.plotly_chart(fig_log, use_container_width=True)

with tab2:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(x=pop_df["fault_distance_km"], y=pop_df["estimated_population"], mode="markers", opacity=0.3, name="Raw"))
    fig_raw.add_trace(go.Scatter(x=lowess_pop[:, 0], y=lowess_pop[:, 1], mode="lines", name="LOWESS", line=dict(color="purple")))
    fig_raw.update_layout(xaxis_title="Distance from Fault (km)", yaxis_title="Estimated Population")
    st.plotly_chart(fig_raw, use_container_width=True)

# === Row 4 ===
st.markdown("---")
# === Row 4: Hexbin-Style Density Plot ===
st.subheader("🔶 Hexbin: Fault Distance vs Population Affected")
st.markdown("**Heatbin (2D Histogram)** of Fault Distance vs Population")

hex_df = df.dropna(subset=["fault_distance_km", "estimated_population"])
# Clip extremely high population values for visibility (optional)
pop_clip_threshold = hex_df["estimated_population"].quantile(0.98)
hex_df = hex_df[hex_df["estimated_population"] <= pop_clip_threshold]

fig_hexbin = px.density_heatmap(
    hex_df,
    x="fault_distance_km",
    y="estimated_population",
    nbinsx=50,
    nbinsy=50,
    color_continuous_scale="YlGnBu",
    labels={
        "fault_distance_km": "Distance from Fault (km)",
        "estimated_population": "Population Affected"
    }
)

fig_hexbin.update_layout(
    xaxis_title="Distance from Fault (km)",
    yaxis_title="Population Affected",
    coloraxis_colorbar=dict(title="Density"),
)

st.plotly_chart(fig_hexbin, use_container_width=True)


# === Row 5 ===
# st.markdown("---")
# st.subheader("Contour: Magnitude vs Population Affected")

# contour_df = df.dropna(subset=["mag", "estimated_population"])
# fig_hexbin = px.density_heatmap(
#     hex_df,
#     x="fault_distance_km",
#     y="estimated_population",
#     nbinsx=50,
#     nbinsy=50,
#     color_continuous_scale="YlGnBu",
#     labels={
#         "fault_distance_km": "Distance from Fault (km)",
#         "estimated_population": "Population Affected"
#     }
# )

# fig_hexbin.update_layout(
#     xaxis_title="Distance from Fault (km)",
#     yaxis_title="Population Affected",
#     coloraxis_colorbar=dict(title="Density"),
# )
# st.plotly_chart(fig_hexbin, use_container_width=True)

# # Optional: add scatter markers
# fig_contour.add_trace(go.Scatter(
#     x=contour_df["mag"],
#     y=contour_df["estimated_population"],
#     mode='markers',
#     marker=dict(size=3, color='rgba(0,0,0,0.3)'),
#     name='Earthquakes'
# ))

# fig_contour.update_layout(
#     xaxis_title="Magnitude",
#     yaxis_title="Population Affected",
#     height=500
# )

# st.plotly_chart(fig_contour, use_container_width=True)
