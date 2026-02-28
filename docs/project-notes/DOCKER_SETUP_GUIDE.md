# Docker Setup Guide for Windows

## Docker Installation (Independent of Conda Environment)

Docker is a system-level application that runs independently of Python conda environments. Once installed, Docker will work with any conda environment, including your `churn_mlops` environment.

## Step-by-Step Installation

### 1. Check System Requirements
- Windows 10 or 11 (64-bit)
- 4GB+ RAM recommended
- Hardware virtualization enabled

### 2. Enable WSL 2 (Required for Docker)
```powershell
# Run PowerShell as Administrator
wsl --install
```

### 3. Download Docker Desktop
1. Visit: https://www.docker.com/products/docker-desktop/
2. Download Docker Desktop for Windows
3. Run the installer

### 4. Install Docker Desktop
- Follow the installation wizard
- Enable WSL 2 backend when prompted
- Restart your computer after installation

### 5. Verify Installation
```powershell
# Open any terminal (doesn't need conda activation)
docker --version
docker-compose --version
```

### 6. Test with Hello World
```powershell
docker run hello-world
```

## Using Docker with Conda Environment

Once Docker is installed, you can use it from any conda environment:

```bash
# Activate your conda environment (optional for Docker commands)
conda activate churn_mlops

# Build and run containers (Docker works regardless of conda env)
docker build -t mlops-churn-prediction -f deployment/docker/Dockerfile .
```

## Containerized Execution Commands

### Build Docker Image
```bash
docker build -t mlops-churn-prediction -f deployment/docker/Dockerfile .
```

### Run Containerized Training
```bash
docker run -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/mlartifacts:/app/mlartifacts mlops-churn-prediction python scripts/run_training.py
```

### Run Complete Pipeline
```bash
docker run -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/mlartifacts:/app/mlartifacts mlops-churn-prediction python run_churn_pipeline.py --model-type xgboost --deploy-to-production
```

## Troubleshooting

### If Docker commands don't work:
1. Ensure Docker Desktop is running
2. Check if WSL 2 is enabled: `wsl --list --verbose`
3. Restart Docker Desktop if needed

### Virtualization Issues:
- Enable virtualization in BIOS/UEFI settings
- Check in Task Manager → Performance → CPU → Virtualization: Enabled

## Benefits of Docker + Conda
- **Conda**: Manages Python dependencies and environments
- **Docker**: Provides containerization and isolation
- **Together**: Perfect combination for reproducible ML workflows

The containerized execution setup is now ready to use once Docker is installed on your system.
