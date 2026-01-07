#!/usr/bin/env python3
"""
Streamlit Application Launcher for Churn Prediction
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    """Main launcher for Streamlit app"""
    parser = argparse.ArgumentParser(description="Launch Streamlit Churn Prediction App")
    parser.add_argument("--port", type=int, default=8501, help="Port to run Streamlit on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ["STREAMLIT_SERVER_PORT"] = str(args.port)
    os.environ["STREAMLIT_SERVER_ADDRESS"] = args.host
    
    # Launch Streamlit
    streamlit_script = Path(__file__).parent.parent / "src" / "deployment" / "streamlit_app.py"
    
    cmd = [
        "streamlit", "run",
        str(streamlit_script),
        "--server.port", str(args.port),
        "--server.address", args.host,
    ]
    
    if args.debug:
        cmd.extend(["--logger.level", "debug"])
    
    print(f"Starting Streamlit app on {args.host}:{args.port}")
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()
