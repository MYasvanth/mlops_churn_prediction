# Setup Instructions

## Prerequisites

- Python 3.8+
- Docker (optional)
- Kubernetes (optional)

## Installation

### 1. Clone the Repository

```
bash
git clone <repository-url>
cd mlops_churn_prediction
```

### 2. Create Virtual Environment

```
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```
bash
pip install -r requirements.txt
pip install -r requirements_dev.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```
env
MLFLOW_TRACKING_URI=<your-mlflow-uri>
DATABASE_URL=<your-database-url>
```

## Running the Project

### Local Development

```
bash
# Run data ingestion
python scripts/ingestion/run_ingestion.py

# Train models
python scripts/training/run_training.py

# Start API server
python -m src.api.main
```

### Using Docker

```
bash
docker-compose -f deployment/docker/docker-compose.yml up
```

## Documentation

This project uses Sphinx for documentation. To build:

```
bash
cd docs
make html
```

The documentation will be generated in `docs/_build/html/`.
