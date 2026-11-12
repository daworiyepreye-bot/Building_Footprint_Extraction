"""
Bayelsa State Digital Addressing System - Phase 1 Backend Server
Serves the web frontend interface and connects user-uploaded GeoTIFF (.tif)
satellite imagery directly to the AI Building Extraction Pipeline.
"""

import os
import sys
import json
import shutil
from flask import Flask, request, jsonify, send_from_directory, send_file, render_template

from building_extraction_pipeline import BuildingExtractionPipeline, generate_sample_bayelsa_tif

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(
    __name__,
    static_folder=FRONTEND_DIR,
    template_folder=FRONTEND_DIR
)

# Initialize extraction pipeline
pipeline = BuildingExtractionPipeline()

LATEST_GEOJSON = os.path.join(OUTPUT_DIR, "student_1_building_footprints.geojson")
LATEST_HTML_MAP = os.path.join(OUTPUT_DIR, "output_footprints_map.html")


def generate_html_map_preview(geojson_path: str, output_html_path: str):
    """Generates a standalone Folium HTML map for download / standalone viewing."""
    try:
        import folium
        import geopandas as gpd

        if not os.path.exists(geojson_path):
            return

        gdf = gpd.read_file(geojson_path)
        if len(gdf) == 0:
            return

        bounds = gdf.total_bounds
        center_lat = (bounds[1] + bounds[3]) / 2.0
        center_lon = (bounds[0] + bounds[2]) / 2.0

        m = folium.Map(location=[center_lat, center_lon], zoom_start=18, tiles="OpenStreetMap")

        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri World Imagery',
            name='Satellite Imagery',
            overlay=False,
            control=True
        ).add_to(m)

        def style_fn(feature):
            return {
                'fillColor': '#ff2a00',
                'color': '#ff2a00',
                'weight': 3,
                'fillOpacity': 0.45
            }

        fields = [f for f in ['building_id', 'confidence_score', 'confidence'] if f in gdf.columns]
        aliases = [f.replace('_', ' ').title() + ':' for f in fields]

        folium.GeoJson(
            gdf,
            name="Extracted Footprints (Student 1)",
            style_function=style_fn,
            tooltip=folium.GeoJsonTooltip(fields=fields, aliases=aliases, localize=True)
        ).add_to(m)

        folium.LayerControl().add_to(m)
        m.save(output_html_path)
    except Exception as e:
        print(f"[WARN] Could not generate Folium map preview: {e}")


@app.route("/", methods=["GET"])
def index():
    """Serves the main frontend UI."""
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/static/<path:filename>", methods=["GET"])
def static_files(filename):
    """Serves CSS, JS, and asset files without aggressive caching."""
    response = send_from_directory(FRONTEND_DIR, filename)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


ALLOWED_EXTENSIONS = (".tif", ".tiff", ".geotiff", ".vrt", ".png", ".jpg", ".jpeg", ".bmp")

@app.route("/api/extract", methods=["POST"])
def extract_from_upload():
    """
    Receives an uploaded satellite raster tile (.tif, .png, .jpg, etc.),
    executes building extraction, strictly validates Master Data Contract 1,
    and returns features + metrics.
    """
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded in form data."}), 400

    uploaded_file = request.files["file"]
    if not uploaded_file.filename:
        return jsonify({"error": "No file selected."}), 400

    filename = uploaded_file.filename
    if not filename.lower().endswith(ALLOWED_EXTENSIONS):
        return jsonify({"error": f"Invalid format. Please upload satellite imagery ({', '.join(ALLOWED_EXTENSIONS)})."}), 400

    conf = float(request.form.get("conf", 0.25))
    tolerance = float(request.form.get("tolerance", 0.00001))

    # Save to uploads directory
    target_path = os.path.join(UPLOAD_DIR, filename)
    uploaded_file.save(target_path)

    # Check for empty virtual file
    file_bytes = os.path.getsize(target_path)
    if file_bytes < 500:
        return jsonify({
            "error": f"The file '{filename}' is an empty virtual header ({file_bytes} bytes, 0x0 pixels) with no imagery data attached. Please upload a real GeoTIFF (.tif) or image file, or click 'Test Sample Tile'."
        }), 400

    try:
        # Run AI extraction pipeline
        result = pipeline.process_tif_to_geojson(
            tif_path=target_path,
            output_geojson_path=LATEST_GEOJSON,
            conf_threshold=conf,
            simplify_tolerance=tolerance
        )

        # Copy to root student_1_building_footprints.geojson for teammate compatibility
        shutil.copy(LATEST_GEOJSON, os.path.join(BASE_DIR, "student_1_building_footprints.geojson"))

        # Generate standalone HTML map
        generate_html_map_preview(LATEST_GEOJSON, LATEST_HTML_MAP)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Building extraction failed: {str(e)}"}), 500


@app.route("/api/sample", methods=["POST"])
def extract_from_sample():
    """
    Synthesizes and runs extraction on the built-in Bayelsa satellite tile (Yenagoa or Amassoma).
    Allows 1-click end-to-end testing without local files.
    """
    conf = float(request.form.get("conf", 0.25))
    tolerance = float(request.form.get("tolerance", 0.00001))
    location = request.form.get("location", "yenagoa").lower()

    filename = f"sample_{location}_tile.tif"
    sample_tif = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(sample_tif):
        generate_sample_bayelsa_tif(sample_tif, location=location)

    try:
        result = pipeline.process_tif_to_geojson(
            tif_path=sample_tif,
            output_geojson_path=LATEST_GEOJSON,
            conf_threshold=conf,
            simplify_tolerance=tolerance
        )

        shutil.copy(LATEST_GEOJSON, os.path.join(BASE_DIR, "student_1_building_footprints.geojson"))
        generate_html_map_preview(LATEST_GEOJSON, LATEST_HTML_MAP)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Sample extraction failed: {str(e)}"}), 500

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Sample extraction failed: {str(e)}"}), 500


@app.route("/api/download/geojson", methods=["GET"])
def download_geojson():
    """Downloads the generated Master Data Contract 1 GeoJSON file."""
    if not os.path.exists(LATEST_GEOJSON):
        return jsonify({"error": "No GeoJSON footprint file has been generated yet."}), 404

    return send_file(
        LATEST_GEOJSON,
        as_attachment=True,
        download_name="student_1_building_footprints.geojson",
        mimetype="application/geo+json"
    )


@app.route("/api/download/map", methods=["GET"])
def download_map():
    """Opens or downloads the standalone Folium interactive HTML map."""
    if not os.path.exists(LATEST_HTML_MAP):
        return jsonify({"error": "No map preview generated yet."}), 404

    return send_file(
        LATEST_HTML_MAP,
        as_attachment=False,
        mimetype="text/html"
    )


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health status and pipeline capabilities."""
    return jsonify({
        "status": "healthy",
        "student": "Student 1 (Daworiye Preye)",
        "role": "Building Footprint Extraction",
        "contract": "Master Data Contract 1",
        "target_consumer": "Student 2 (Addressing Engine)",
        "yolo_enabled": pipeline.yolo_enabled
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"\n========================================================")
    print(f"🚀 Bayelsa Building Extraction Portal Started!")
    print(f"   Open your browser at: http://127.0.0.1:{port}")
    print(f"========================================================\n")
    app.run(host="127.0.0.1", port=port, debug=False)
