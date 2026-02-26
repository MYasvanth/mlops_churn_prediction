Monitoring Guide
===============

Overview
--------

The monitoring system tracks model performance, data drift, and system health in production.

Components
----------

Data Drift Monitoring
~~~~~~~~~~~~~~~~~~~~

**Purpose**: Detect changes in input data distribution

**Implementation**: Evidently AI integration

.. code-block:: bash

   # Run drift detection
   python scripts/monitoring/run_data_drift_monitor.py

   # Automated monitoring
   python scripts/monitoring/run_data_drift_monitor.py --continuous

**Metrics Tracked**:

* Feature drift scores
* Target drift detection
* Statistical tests (KS, PSI)
* Distribution comparisons

Model Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Track model accuracy and performance metrics

**Metrics**:

* Accuracy, Precision, Recall, F1-Score
* Prediction latency
* Throughput (requests/second)
* Error rates

.. code-block:: python

   # Access performance metrics
   import requests
   
   response = requests.get("http://localhost:8000/monitoring/metrics")
   metrics = response.json()

System Health Monitoring
~~~~~~~~~~~~~~~~~~~~~~~

**Purpose**: Monitor infrastructure and service health

**Health Checks**:

* API endpoint availability
* Model loading status
* Database connectivity
* Resource utilization

Alert System
-----------

Configuration
~~~~~~~~~~~~

**File**: ``configs/monitoring/alerts.yaml``

.. code-block:: yaml

   alerts:
     data_drift:
       threshold: 0.1
       notification: email
       recipients: ["admin@company.com"]
     
     performance_degradation:
       accuracy_threshold: 0.75
       notification: slack
       webhook_url: "https://hooks.slack.com/..."

Alert Types
~~~~~~~~~~

1. **Data Drift Alerts**
   - Triggered when drift score exceeds threshold
   - Includes drift report and affected features

2. **Performance Alerts**
   - Model accuracy below threshold
   - High prediction latency
   - Service unavailability

3. **System Alerts**
   - High memory/CPU usage
   - Disk space warnings
   - Database connection issues

Dashboards
----------

Streamlit Dashboard
~~~~~~~~~~~~~~~~~

**URL**: http://localhost:8501

**Features**:

* Real-time predictions
* Model performance metrics
* Data drift visualization
* System health status

.. code-block:: bash

   streamlit run src/deployment/streamlit_app.py --server.port 8501

MLflow Dashboard
~~~~~~~~~~~~~~~

**URL**: http://localhost:5000

**Features**:

* Experiment tracking
* Model registry
* Performance comparison
* Artifact management

Grafana Integration
~~~~~~~~~~~~~~~~~

**Setup**:

.. code-block:: bash

   # Start Grafana
   docker run -d -p 3000:3000 grafana/grafana

   # Import dashboard
   # Use monitoring/dashboards/grafana.json

**Metrics Displayed**:

* API response times
* Request volumes
* Error rates
* Model performance trends

Logging
-------

Log Configuration
~~~~~~~~~~~~~~~

**File**: ``configs/monitoring/logging.yaml``

.. code-block:: yaml

   logging:
     level: INFO
     format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
     handlers:
       - file: logs/mlops.log
       - console: true

Log Analysis
~~~~~~~~~~~

.. code-block:: bash

   # View recent logs
   tail -f logs/mlops_$(date +%Y%m%d).log

   # Search for errors
   grep "ERROR" logs/mlops_*.log

   # Monitor API requests
   grep "POST /predict" logs/mlops_*.log

Automated Monitoring
------------------

Continuous Monitoring
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Start continuous monitoring
   python scripts/monitoring/run_monitoring.py --mode continuous

   # Schedule with cron
   # Add to crontab:
   # 0 */6 * * * /path/to/python scripts/monitoring/run_monitoring.py --mode single

Report Generation
~~~~~~~~~~~~~~~

**Automated Reports**:

* Daily performance summaries
* Weekly drift analysis
* Monthly model evaluation

.. code-block:: bash

   # Generate reports
   python scripts/monitoring/generate_reports.py --period daily

Best Practices
-------------

1. **Baseline Establishment**: Set performance baselines during initial deployment
2. **Threshold Tuning**: Adjust alert thresholds based on business requirements
3. **Regular Reviews**: Schedule periodic monitoring system reviews
4. **Documentation**: Keep monitoring procedures documented and updated