# Render Deployment - Using Existing Code

## 🚀 Deploy Your MLOps Pipeline

### 1. Push to GitHub
```bash
git add .
git commit -m "Deploy to Render"
git push origin main
```

### 2. Create Render Services

**FastAPI Service:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `python scripts/monitoring/run_fastapi_server.py`

**Streamlit Dashboard:**
- Build Command: `pip install -r requirements.txt`
- Start Command: `streamlit run src/deployment/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

### 3. Environment Variables
Set `PYTHONPATH=.` for both services.

### 4. Test Deployment
- API: `https://your-api.onrender.com/health`
- Dashboard: `https://your-dashboard.onrender.com`

That's it! Your existing MLOps pipeline is deployment-ready.