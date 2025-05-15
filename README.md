# 🌍 Earthquake Impact and Population Risk Visualization

This project analyzes global earthquake data and visualizes population exposure risk using seismic shaking models and global population datasets. The accompanying NeurIPS-style report summarizes our findings, methodology, and visualizations.

## 📊 Datasets Used

### 1. **USGS Earthquake Catalog**
- **Source**: [USGS API](https://earthquake.usgs.gov/fdsnws/event/1/)
- **How to download**:  
  ```bash
  python src/fetch_earthquake_data.py

### 2. WorldPop Global Population (2020)
- **Source**: WorldPop GeoData Portal

- **Manual option**:
   Download GeoTIFFs for relevant countries → place in data/population/

- **Scripted (optional)**:
  ```bash
  python src/fetch_population_data.py

## ⚙️ Setup & Execution

### 1. Clone and Setup Environment

```bash
git clone https://github.com/yourusername/earthquake-visualization.git
cd earthquake-visualization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # (Windows: venv\Scripts\activate)

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Data Collection
```bash
python src/fetch_earthquake_data.py
python src/fetch_population_data.py
```

### 3. Generate Visualizations
```bash
python src/plot_earthquake_map.py
```
### 4. Estimate Population Impact
```bash
python src/estimate_impact.py
```
### 5. Launch the Interactive Dashboard (Streamlit)
After all data has been collected and processed, run the dashboard using Streamlit:

```bash
streamlit run test_app.py
```
This will launch the Earthquake Impact Visualization Dashboard in your browser, where you can interactively explore:

- Earthquake locations

- Magnitude and depth statistics

- Population exposure estimates

- Shockwave and terrain-based animations (if implemented)
  
## 🧪 Dependencies
List of libraries (in requirements.txt):

```bash
pandas
numpy
matplotlib
seaborn
geopandas
rasterio
pydeck
plotly
shapely
requests
```
Install with:

```bash
pip install -r requirements.txt
```
## 👨‍💻 Contributors
Suriya Kasiyalan Siva – Project Lead, Data Visualization, Report Writing

## 📜 License
This project is intended for academic use only and follows the Northeastern University academic integrity policies.
