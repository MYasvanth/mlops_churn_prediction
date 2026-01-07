#!/usr/bin/env python3
"""
Optuna Dashboard Launcher for Hyperparameter Monitoring
"""

import argparse
import os
import sys
from pathlib import Path

def main():
    """Main launcher for Optuna dashboard"""
    parser = argparse.ArgumentParser(description="Launch Optuna Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to run dashboard on")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--storage", default="sqlite:///optuna_studies.db", 
                       help="Optuna storage URL")
    
    args = parser.parse_args()
    
    cmd = [
        "optuna-dashboard",
        "--port", str(args.port),
        "--host", args.host,
        args.storage
    ]
    
    print(f"Starting Optuna dashboard on {args.host}:{args.port}")
    print(f"Storage: {args.storage}")
    os.system(" ".join(cmd))

if __name__ == "__main__":
    main()
