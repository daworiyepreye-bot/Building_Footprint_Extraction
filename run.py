"""
Interactive Map Visualizer for Student 1 Building Footprints
Renders extracted GeoJSON building footprints onto OpenStreetMap / Satellite basemap.
Saves map to 'output_footprints_map.html' for local browsing, and displays in Jupyter/Colab.
"""

import os
import sys
import folium
import geopandas as gpd

# Prioritize local path, CLI argument, or fallback to Colab path
if len(sys.argv) > 1:
    geojson_path = sys.argv[1]
elif os.path.exists('student_1_building_footprints.geojson'):
    geojson_path = 'student_1_building_footprints.geojson'
elif os.path.exists('/content/student_1_building_footprints.geojson'):
    geojson_path = '/content/student_1_building_footprints.geojson'
else:
    geojson_path = 'student_1_building_footprints.geojson'

if not os.path.exists(geojson_path):
    print(f"[ERROR] Handoff file not found at '{geojson_path}'. Please run building_extraction_pipeline.py first.")
else:
    print(f"[INFO] Loading building footprints from: '{geojson_path}'...")
    gdf = gpd.read_file(geojson_path)
    print(f"[INFO] Loaded {len(gdf)} footprint polygons.")

    # Calculate geographic center
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    center_lat = (bounds[1] + bounds[3]) / 2.0
    center_lon = (bounds[0] + bounds[2]) / 2.0

    # Initialize Folium Map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="OpenStreetMap")

    # Add Esri Satellite Imagery as an alternative layer
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri World Imagery',
        name='Satellite Imagery',
        overlay=False,
        control=True
    ).add_to(m)

    # Styling for building footprints
    def style_function(feature):
        return {
            'fillColor': '#ff2a00',
            'color': '#ff2a00',
            'weight': 3,
            'fillOpacity': 0.45
        }

    # Add GeoJSON layer with interactive tooltips
    fields = [f for f in ['building_id', 'confidence_score', 'confidence'] if f in gdf.columns]
    aliases = [f.replace('_', ' ').title() + ':' for f in fields]

    folium.GeoJson(
        gdf,
        name="Extracted Footprints (Student 1)",
        style_function=style_function,
        tooltip=folium.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            localize=True
        )
    ).add_to(m)

    # Layer control panel
    folium.LayerControl().add_to(m)

    # Save to HTML file
    output_html = "output_footprints_map.html"
    m.save(output_html)
    print(f"[SUCCESS] Interactive map successfully saved to: '{output_html}'")

    # Try display if inside notebook
    try:
        from IPython.display import display
        display(m)
    except (ImportError, NameError):
        pass
