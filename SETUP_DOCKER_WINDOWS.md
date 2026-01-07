# Docker Setup for Windows

## Prerequisites
- Windows 10 or Windows 11 (64-bit)
- At least 4GB RAM (8GB recommended)
- Hardware Virtualization enabled in BIOS

## Step 1: Enable WSL 2 (Windows Subsystem for Linux)

### Option A: Using PowerShell (Admin)
```powershell
# Run PowerShell as Administrator and execute:
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

### Option B: Using Windows Features GUI
1. Press `Win + R`, type `optionalfeatures.exe` and press Enter
2. Enable:
   - Windows Subsystem for Linux
   - Virtual Machine Platform
3. Restart your computer

## Step 2: Install WSL 2 Linux Kernel
Download and install from: https://aka.ms/wsl2kernel

## Step 3: Install Docker Desktop for Windows
1. Download Docker Desktop from: https://desktop.docker.com/win/main/amd64/Docker%20Desktop%20Installer.exe
2. Run the installer
3. Follow the installation wizard
4. Restart your computer when prompted

## Step 4: Configure Docker Desktop
1. Open Docker Desktop
2. Go to Settings → General
   - Enable "Use the WSL 2 based engine"
3. Go to Settings → Resources → WSL Integration
   - Enable integration with your default WSL distro
4. Apply & Restart

## Step 5: Verify Installation
```powershell
# Open PowerShell and run:
docker --version
docker-compose --version
docker run hello-world
```

## Step 6: Test MLOps Containerized Execution
```powershell
# Build the Docker image
docker build -t mlops-churn-prediction -f deployment/docker/Dockerfile .

# Run containerized training
docker run -v ${PWD}/data:/app/data -v ${PWD}/models:/app/models -v ${PWD}/mlartifacts:/app/mlartifacts mlops-churn-prediction python scripts/run_training.py

# Or use the automated script
python scripts/containerized_execution.py
```

## Troubleshooting

### Common Issues:
1. **Docker not starting**: Check if virtualization is enabled in BIOS
2. **WSL 2 not working**: Run `wsl --set-default-version 2`
3. **Permission errors**: Run Docker Desktop as Administrator

### Enable Virtualization in BIOS:
1. Restart computer and enter BIOS/UEFI settings (usually F2, F10, or Del key)
2. Enable Intel VT-x or AMD-V virtualization
3. Save and exit

## Alternative: Use Docker in WSL 2 Directly
If Docker Desktop has issues, you can install Docker directly in WSL 2:

```bash
# Inside WSL 2 (Ubuntu)
sudo apt update
sudo apt install docker.io
sudo service docker start
```

## Resources
- Docker Desktop Documentation: https://docs.docker.com/desktop/
- WSL 2 Installation Guide: https://docs.microsoft.com/en-us/windows/wsl/install
- MLOps with Docker: https://mlops-guide.github.io/
