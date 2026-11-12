"""
Google Colab Sync Helper for Student 1 Pipeline
Writes the complete updated building_extraction_pipeline.py into Colab's /content directory.
"""

import os
import shutil

src_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "building_extraction_pipeline.py")

with open(src_file, "r", encoding="utf-8") as f:
    code = f.read()

colab_dest = "/content/building_extraction_pipeline.py"
try:
    with open(colab_dest, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[SUCCESS] Updated pipeline written to '{colab_dest}' for Colab environment.")
except Exception:
    # If running locally outside Colab
    print(f"[INFO] Code is up to date locally at '{src_file}'.")
