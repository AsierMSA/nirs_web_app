"""
Application entry point for running the NIRS Analysis backend server.
"""
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    print("=" * 60)
    print("🧠 Starting NIRS Analysis Backend Server")
    print(f"🚀 Server running on: http://localhost:{port}")
    print("📡 Ready to receive requests from React frontend")
    print("=" * 60)

    app.run(host=host, port=port, debug=debug)
