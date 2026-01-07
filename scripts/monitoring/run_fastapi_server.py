#!/usr/bin/env python3
"""
Script to start the FastAPI server for churn prediction model deployment.
This separates the API layer from the model serving logic.
"""

import uvicorn
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """Start the FastAPI server."""
    print("Starting FastAPI Server for Churn Prediction...")
    print("Project Root:", project_root)
    print("API Documentation: http://localhost:8000/docs")
    print("Health Check: http://localhost:8000/health")
    print("Press Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "src.deployment.api_endpoints:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
