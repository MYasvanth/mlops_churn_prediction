Usage Guide
===========

Quick Start
-----------

Training Models
~~~~~~~~~~~~~~

.. code-block:: bash

   # Basic training
   python run_churn_pipeline.py --model-type xgboost

   # With hyperparameter optimization
   python run_churn_pipeline.py --model-type lightgbm --hyperparameter-optimization --n-trials 100

   # Deploy to production
   python run_churn_pipeline.py --model-type xgboost --deploy-to-production

Starting Services
~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start API server
   python scripts/monitoring/run_fastapi_server.py

   # Launch dashboard
   streamlit run src/deployment/streamlit_app.py --server.port 8501

   # View experiment tracking
   mlflow ui --port 5000

Making Predictions
-----------------

API Requests
~~~~~~~~~~~

.. code-block:: bash

   curl -X POST "http://localhost:8000/predict" \
        -H "Content-Type: application/json" \
        -d '{
          "features": [0.5, 0.3, 0.8, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.8, 
                       0.3, 0.5, 0.2, 0.9, 0.1, 0.7, 0.4, 0.6, 0.8, 0.3],
          "model_id": "xgboost_20250822_013133",
          "stage": "staging"
        }'

Python Client
~~~~~~~~~~~~

.. code-block:: python

   import requests

   # Single prediction
   response = requests.post("http://localhost:8000/predict", json={
       "features": [0.5] * 20,
       "model_id": "xgboost_20250822_013133",
       "stage": "staging"
   })
   
   result = response.json()
   print(f"Prediction: {result['prediction']}")
   print(f"Probability: {result['probability']}")

Monitoring
----------

Data Drift Detection
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run drift detection
   python scripts/monitoring/run_data_drift_monitor.py

   # View drift reports
   ls reports/drift_reports/

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~

Access monitoring dashboards:

* **MLflow UI**: http://localhost:5000
* **Streamlit Dashboard**: http://localhost:8501
* **Optuna Dashboard**: http://localhost:8080
* **API Documentation**: http://localhost:8000/docs

Testing
-------

Running Tests
~~~~~~~~~~~~

.. code-block:: bash

   # Run all tests
   pytest tests/

   # Run specific test module
   pytest tests/test_models/

   # Run with coverage
   pytest --cov=src --cov-report=html

Configuration
------------

Model Configuration
~~~~~~~~~~~~~~~~~

Edit model parameters in ``configs/model/`` directory:

.. code-block:: yaml

   xgboost:
     n_estimators: 100
     max_depth: 6
     learning_rate: 0.1
     subsample: 0.8

Pipeline Configuration
~~~~~~~~~~~~~~~~~~~~

Modify pipeline settings in ``params.yaml``:

.. code-block:: yaml

   data_ingestion:
     source_path: "data/raw/churn_data.csv"
     validation_rules: "configs/data/validation_rules.yaml"

   training:
     test_size: 0.2
     random_state: 42
     cv_folds: 5