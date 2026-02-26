# CI/CD Pipeline

## Pipeline Overview

```
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│  Code   │──▶│  Build  │──▶│  Test   │──▶│ Deploy  │──▶│ Monitor │
│  Commit │   │         │   │         │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘
```

## CI/CD Tools

| Stage | Tool |
|-------|------|
| Source Control | GitHub |
| CI/CD | GitHub Actions |
| Container Registry | Docker Hub / ECR |
| Orchestration | Kubernetes |
| Infrastructure | Terraform |

## GitHub Actions Workflow

### Training Pipeline

```
yaml
name: Model Training
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements_dev.txt
      
      - name: Run data ingestion
        run: python scripts/ingestion/run_ingestion.py
      
      - name: Train model
        run: python scripts/training/run_training.py
      
      - name: Evaluate model
        run: python scripts/evaluation/run_evaluation.py
      
      - name: Register model
        if: github.ref == 'refs/heads/main'
        run: mlflow models register
      
      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: model-artifacts
          path: models/production/
```

### Deployment Pipeline

```
yaml
name: Deploy to Production
on:
  push:
    branches: [main]
    paths: ['models/**']
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: |
          docker build -t churn-api:${{ github.sha }} .
      
      - name: Push to registry
        run: |
          docker push registry.example.com/churn-api:${{ github.sha }}
      
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/churn-api \
            api=registry.example.com/churn-api:${{ github.sha }}
```

## Stages

### 1. Code Quality
- Linting (flake8, pylint)
- Code formatting (black)
- Static analysis (SonarQube)

### 2. Testing
- Unit tests (pytest)
- Integration tests
- Model validation tests

### 3. Build
- Docker image build
- Multi-architecture build (AMD64/ARM64)
- Image scanning (Trivy)

### 4. Deploy
- Staging deployment
- Smoke tests
- Production deployment
- Canary release

### 5. Monitor
- Health checks
- Performance monitoring
- Error tracking

## Environments

| Environment | Purpose | Branch |
|-------------|---------|--------|
| Development | Testing | feature/* |
| Staging | QA | develop |
| Production | Live | main |

## Rollback Strategy

1. **Automatic Rollback**: Triggered on:
   - Health check failure
   - Error rate > 5%
   - Latency p99 > 500ms

2. **Manual Rollback**:
   
```
bash
   kubectl rollout undo deployment/churn-api
   
```

## Secrets Management

- **GitHub Secrets**: API keys, tokens
- **Vault**: Production credentials
- **AWS Secrets Manager**: Database passwords
