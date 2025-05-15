
import streamlit as st
import pandas as pd
import plotly.express as px

@st.cache_data
def load_data():
    return pd.read_csv("earthquakes_with_population.csv")  # Assumes 'country_iso3', 'time', 'estimated_population', 'mag' columns

df = load_data()
df["time"] = pd.to_datetime(df["time"], errors="coerce")
df = df.dropna(subset=["country_iso3", "estimated_population", "time"])

# st.set_page_config(page_title="Country-wise Earthquake Impact", layout="wide")
st.title("📌 Country-wise Earthquake Insights")

# Layout
col1, col2 = st.columns(2)

# === Plot 1: Country-wise Total Estimated Population Impact (Bar)
with col1:
    st.subheader("👥 Total Population Affected per Country")
    agg_country = df.groupby("country_iso3")["estimated_population"].sum().reset_index()
    agg_country = agg_country.sort_values("estimated_population", ascending=False).head(15)
    fig1 = px.bar(agg_country, x="country_iso3", y="estimated_population",
                  labels={"country_iso3": "Country", "estimated_population": "Population Affected"},
                  color="estimated_population", color_continuous_scale="Viridis")
    st.plotly_chart(fig1, use_container_width=True)

# === Plot 2: Time-Series of Population Affected
with col2:
    # st.subheader("📈 Population Impact Over Time")
    # time_series = df.set_index("time").resample("M")["estimated_population"].sum().reset_index()
    # fig2 = px.line(time_series, x="time", y="estimated_population",
    #                labels={"time": "Date", "estimated_population": "People Affected"},
    #                title="Monthly Population Impact from Earthquakes")
    # st.plotly_chart(fig2, use_container_width=True)
    # st.subheader("📈 Total Population Affected by Country (Within Earthquake Radius)")

    # from gmpe_radius_estimator import estimate_radius_pga_gmpe

    # # Step 1: Keep only rows with valid values
    # df_radius = df.dropna(subset=["country_iso3", "estimated_population", "mag", "latitude", "longitude"]).copy()

    # # Step 2: Compute GMPE-based shaking radius
    # df_radius["impact_radius_km"] = df_radius.apply(
    #     lambda row: estimate_radius_pga_gmpe(row["mag"], row["latitude"], row["longitude"]), axis=1
    # )

    # # 📝 At this point, population is already assumed to be within the radius — no need to filter again

    # # Step 3: Aggregate population affected per country
    # country_radius_impact = df_radius.groupby("country_iso3")["estimated_population"].sum().reset_index()
    # country_radius_impact = country_radius_impact.sort_values("estimated_population", ascending=False)

    # # Step 4: Plot
    # fig_country = px.bar(
    #     country_radius_impact,
    #     x="country_iso3", 
    #     y="estimated_population",
    #     labels={"country_iso3": "Country", "estimated_population": "Total Population Affected (Within Radius)"},
    #     title="Total Estimated Population Affected per Country (Within Earthquake Radius)",
    #     color="estimated_population",
    #     color_continuous_scale="Reds"
    # )
    # fig_country.update_layout(xaxis_tickangle=45)
    # st.plotly_chart(fig_country, use_container_width=True)





    # st.subheader("🗺️ Choropleth Map: Country-Level Population Exposure")

    # # Make sure 'country_iso3' exists and clean
    # choropleth_df = df.dropna(subset=["country_iso3", "estimated_population"])
    # country_iso3_sum = choropleth_df.groupby("country_iso3")["estimated_population"].sum().reset_index()

    # # Choropleth map using ISO3 codes
    # fig_map = px.choropleth(
    #     country_iso3_sum,
    #     locations="country_iso3",
    #     color="estimated_population",
    #     hover_name="country_iso3",
    #     color_continuous_scale="Reds",
    #     labels={"estimated_population": "Population Affected"},
    #     title="Total Population Affected by Earthquakes (Per Country)"
    # )
    # fig_map.update_geos(showcoastlines=True, showcountries=True)
    # st.plotly_chart(fig_map, use_container_width=True)



# === Plot 3: Magnitude vs Population Heatmap
    st.subheader("🔥 Heatmap: Magnitude vs Population Affected")
    heat_df = df[df["estimated_population"] > 0].copy()
    heat_df["log_population"] = heat_df["estimated_population"].apply(lambda x: max(1, x))
    fig3 = px.density_heatmap(heat_df, x="mag", y="log_population", nbinsx=30, nbinsy=40,
                            labels={"mag": "Magnitude", "log_population": "Log Population"},
                            color_continuous_scale="Magma")
    st.plotly_chart(fig3, use_container_width=True)






# ===== Plot 1: Magnitude vs Population Affected =====
scatter_df = df.dropna(subset=["mag", "estimated_population", "depth"])

# Clip negative depth values for safe size rendering
scatter_df["size_safe"] = scatter_df["depth"].clip(lower=0)

fig = px.scatter(
    scatter_df,
    x="mag",
    y="estimated_population",
    size="size_safe",  # Now this column exists
    color="mag",
    labels={
        "mag": "Magnitude",
        "estimated_population": "Population Affected",
        "size_safe": "Depth (km)"
    },
    title="Magnitude vs Population Affected",
    color_continuous_scale="Turbo"
)
st.plotly_chart(fig, use_container_width=True)


# # ===== Plot 2: Country Total Pop vs Affected Pop =====
# # Assume country population data from tile overlay
# pop_df = pd.read_csv("total_population_by_country_2024.csv")
# df = df.merge(pop_df, on="country_iso3", how="left")
# country_agg = df.groupby("country_iso3").agg({
#     "estimated_population": "sum",
#     "total_population": "first"  # Assuming one value per country
# }).dropna()

# # Calculate exposure percentage
# country_agg["pct_exposed"] = country_agg["estimated_population"] / country_agg["total_population"]
# country_agg = country_agg.reset_index()

# # Scatter plot
# scatter2 = px.scatter(
#     country_agg,
#     x="total_population", y="estimated_population",
#     size="pct_exposed",
#     color="pct_exposed",
#     hover_name="country_iso3",
#     labels={
#         "total_population": "Total Country Population",
#         "estimated_population": "Affected Population",
#         "pct_exposed": "% Population Exposed"
#     },
#     title="🌍 Country Total Pop vs Affected Pop (color = % exposed)",
#     color_continuous_scale="Reds",
#     size_max=30,
#     log_x=True, log_y=True
# )

# scatter2.update_layout(margin=dict(l=20, r=20, t=60, b=20))
# st.plotly_chart(scatter2, use_container_width=True)

# ===== Plot 3: Top 5 Earthquakes Table per Country =====
# Ensure necessary columns are present
st.subheader("Top 5 Earthquakes by Population Affected")

top5_table = df.sort_values("estimated_population", ascending=False).head(5)
top5_table = top5_table[["country_iso3", "place", "mag", "time", "estimated_population"]]

st.dataframe(
    top5_table.style.format({
        "mag": "{:.1f}",
        "estimated_population": "{:,.0f}"
    }),
    use_container_width=True
)


# Optional: Data Preview
st.subheader("📄 Data Preview")
st.dataframe(df[["time", "place", "country_iso3", "mag", "estimated_population"]].sample(10))