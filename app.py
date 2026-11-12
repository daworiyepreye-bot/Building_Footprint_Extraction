"""
Application launcher for Bayelsa Building Extraction Portal
Runs the Flask application defined in server.py
"""

from server import app

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    print(f"\n[SERVER] Starting Student 1 Web Upload Portal on http://127.0.0.1:{port}...")
    app.run(host="127.0.0.1", port=port, debug=False)
