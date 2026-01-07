#!/usr/bin/env python3
"""
Start only the Streamlit dashboard for troubleshooting
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

def start_streamlit():
    """Start Streamlit dashboard with detailed logging"""
    try:
        print("🚀 Starting Streamlit dashboard...")
        
        # Check if port is available
        if not check_port_available(8501):
            print("❌ Port 8501 is already in use")
            return None
        
        cmd = [
            "streamlit", "run", 
            "src/deployment/streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--logger.level", "debug"
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
            print("✅ Streamlit dashboard started successfully")
            print("🌐 Access at: http://localhost:8501")
            return process
        else:
            print(f"❌ Streamlit failed to start. Return code: {process.returncode}")
            stdout, stderr = process.communicate()
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Streamlit: {str(e)}")
        return None

def check_streamlit_health():
    """Check if Streamlit is responding"""
    import requests
    try:
        response = requests.get("http://localhost:8501", timeout=5)
        if response.status_code == 200:
            print("✅ Streamlit dashboard is responding")
            return True
        else:
            print(f"❌ Streamlit responded with status: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Streamlit health check failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Streamlit Dashboard Troubleshooter")
    print("=" * 60)
    
    # Check port availability
    if not check_port_available(8501):
        print("⚠️  Port 8501 is in use. Trying to identify the process...")
        try:
            # Try to find what's using port 8501
            import psutil
            for conn in psutil.net_connections():
                if conn.laddr.port == 8501 and conn.status == 'LISTEN':
                    print(f"Port 8501 is used by PID {conn.pid}")
                    break
        except ImportError:
            print("Install psutil to see which process is using the port")
    
    # Start Streamlit
    process = start_streamlit()
    
    if process:
        print("\n🔄 Waiting for Streamlit to initialize...")
        time.sleep(3)
        
        # Check health
        if check_streamlit_health():
            print("\n🎉 Streamlit dashboard is working correctly!")
            print("Press Ctrl+C to stop the dashboard")
            
            try:
                # Keep the process running
                process.wait()
            except KeyboardInterrupt:
                print("\n🛑 Stopping Streamlit dashboard...")
                process.terminate()
                process.wait()
                print("✅ Streamlit stopped")
        else:
            print("\n❌ Streamlit started but not responding")
            process.terminate()
            sys.exit(1)
    else:
        print("\n❌ Failed to start Streamlit dashboard")
        sys.exit(1)
