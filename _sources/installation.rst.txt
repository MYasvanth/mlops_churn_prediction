Installation Guide
=================

System Requirements
------------------

* Python 3.8+
* Docker (optional)
* Kubernetes (optional)
* 4GB+ RAM
* 10GB+ disk space

Local Installation
-----------------

1. Clone Repository
~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   git clone <repository-url>
   cd mlops_churn_prediction

2. Create Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate

3. Install Dependencies
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   pip install -r requirements.txt

4. Initialize Services
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Initialize ZenML
   zenml init
   
   # Start MLflow
   mlflow server --host 0.0.0.0 --port 5000

5. Verify Installation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python scripts/deployment/final_deployment_verification.py

Docker Installation
------------------

1. Build Image
~~~~~~~~~~~~~

.. code-block:: bash

   docker build -t mlops-churn .

2. Run Container
~~~~~~~~~~~~~~~

.. code-block:: bash

   docker run -p 8000:8000 -p 5000:5000 mlops-churn

Kubernetes Installation
----------------------

1. Apply Manifests
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   kubectl apply -f deployment/kubernetes/

2. Check Status
~~~~~~~~~~~~~~

.. code-block:: bash

   kubectl get pods -l app=mlops-churn

Configuration
------------

Environment Variables
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   export MLFLOW_TRACKING_URI=http://localhost:5000
   export MODEL_STAGE=staging
   export API_HOST=0.0.0.0
   export API_PORT=8000

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

* **Import Errors**: Ensure all dependencies are installed
* **Port Conflicts**: Change ports in configuration files
* **Permission Errors**: Check file permissions and user access