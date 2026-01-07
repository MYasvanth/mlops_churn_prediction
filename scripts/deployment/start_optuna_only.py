#!/usr/bin/env python3
"""
Start only the Optuna dashboard for troubleshooting
"""

import subprocess
import time
import sys
import socket

def check_port_available(port):
    """Check if a port is available"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    return result != 0  # 0 means port is in use

def start_optuna():
    """Start Optuna dashboard with detailed logging"""
    try:
        print("🚀 Starting Optuna dashboard...")
        
        # Check if port is available
        if not check_port_available(8080):
            print("❌ Port 8080 is already in use")
            return None
        
        # Check if the database file exists
        import os
        if not os.path.exists("optuna_studies.db"):
            print("⚠️  Optuna database file not found. Creating empty database...")
            # Create an empty SQLite database for Optuna
            import sqlite3
            conn = sqlite3.connect("optuna_studies.db")
            conn.close()
        
        cmd = [
            "optuna-dashboard",
            "--port", "8080",
            "--host", "0.0.0.0",
            "sqlite:///optuna_studies.db"
        ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Optuna dashboard started successfully")
            print("🌐 Access at: http://localhost:8080")
            return process
        else:
            print(f"❌ Optuna failed to start. Return code: {process.returncode}")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Optuna: {str(e)}")
        return None

def check_optuna_health():
    """Check if Optuna is responding"""
    import requests
    try:
        response = requests.get("http://localhost:8080", timeout=5)
        if response.status_code == 200:
            print("✅ Optuna dashboard is responding")
            return True
        else:
            print(f"❌ Optuna responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Optuna health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Optuna Dashboard Troubleshooter")
    print("=" * 60)
    
    # Check port availability
    if not check_port_available(8080):
        print("⚠️  Port 8080 is in use. Trying to identify the process...")
        try:
            # Try to find what's using port 8080
            import psutil
            for conn in psutil.net_connections():
                if conn.laddr.port == 8080 and conn.status == 'LISTEN':
                    print(f"Port 8080 is used by PID {conn.pid}")
                    break
        except ImportError:
            print("Install psutil to see which process is using the port")
    
    # Start Optuna
    process = start_optuna()
    
    if process:
        print("\n🔄 Waiting for Optuna to initialize...")
        time.sleep(3)
        
        # Check health
        if check_optuna_health():
            print("\n🎉 Optuna dashboard is working correctly!")
            print("Press Ctrl+C to stop the dashboard")
            
            try:
                # Keep the process running
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping Optuna dashboard...")
                process.terminate()
                process.wait()
                print("✅ Optuna stopped")
        else:
            print("\n❌ Optuna started but not responding")
            process.terminate()
            sys.exit(1)
    else:
        print("\n❌ Failed to start Optuna dashboard")
        sys.exit(1)
