"""
Streamlit Application for Churn Prediction Model Deployment
Provides interactive model prediction and monitoring interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import json
import joblib
import mlflow
import optuna
from datetime import datetime

# Add project root to sys.path for imports
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.models.unified_model_registry_fixed import UnifiedModelRegistry
from src.utils.logger import get_logger
from src.utils.config_loader import load_config
from src.monitoring.optuna_monitor import OptunaMonitor
from src.data.data_loader import DataLoader

logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

class ChurnPredictionApp:
    def __init__(self):
        self.config = load_config("configs/model/unified_model_config.yaml")
        self.registry = UnifiedModelRegistry("configs/model/unified_model_config.yaml")
        self.optuna_monitor = OptunaMonitor()
        self.data_loader = DataLoader("configs/data/local.yaml")
        # Load encoders
        try:
            self.data_loader.load_encoders("models/encoders/")
            logger.info("Successfully loaded label encoders")
        except Exception as e:
            logger.error(f"Error loading encoders: {e}")
        
    def run(self):
        """Main application runner"""
        st.markdown('<h1 class="main-header">🎯 Churn Prediction Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar navigation
        page = st.sidebar.selectbox(
            "Navigation",
            ["Model Prediction", "Performance Monitoring", "Optuna Dashboard", "Data Analysis"]
        )
        
        if page == "Model Prediction":
            self.show_prediction_page()
        elif page == "Performance Monitoring":
            self.show_monitoring_page()
        elif page == "Optuna Dashboard":
            self.show_optuna_dashboard()
        elif page == "Data Analysis":
            self.show_data_analysis()
    
    def show_prediction_page(self):
        """Model prediction interface"""
        st.header("🔮 Model Prediction")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Input Customer Data")
            
            # Create input form
            with st.form("prediction_form"):
                # Customer demographics
                col_a, col_b = st.columns(2)
                
                with col_a:
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
                    partner = st.selectbox("Partner", ["No", "Yes"])
                    dependents = st.selectbox("Dependents", ["No", "Yes"])
                
                with col_b:
                    tenure = st.slider("Tenure (months)", 1, 72, 24)
                    phone_service = st.selectbox("Phone Service", ["No", "Yes"])
                    multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
                    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                
                # Service information
                col_c, col_d = st.columns(2)
                
                with col_c:
                    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
                    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
                    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
                    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
                
                with col_d:
                    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
                    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
                    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
                    paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
                
                # Payment information
                payment_method = st.selectbox("Payment Method", [
                    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
                total_charges = st.number_input("Total Charges", 0.0, 10000.0, 2000.0)
                
                submitted = st.form_submit_button("Predict Churn", use_container_width=True)
                
                if submitted:
                    # Prepare input data
                    input_data = {
                        'gender': gender,
                        'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
                        'Partner': partner,
                        'Dependents': dependents,
                        'tenure': tenure,
                        'PhoneService': phone_service,
                        'MultipleLines': multiple_lines,
                        'InternetService': internet_service,
                        'OnlineSecurity': online_security,
                        'OnlineBackup': online_backup,
                        'DeviceProtection': device_protection,
                        'TechSupport': tech_support,
                        'StreamingTV': streaming_tv,
                        'StreamingMovies': streaming_movies,
                        'Contract': contract,
                        'PaperlessBilling': paperless_billing,
                        'PaymentMethod': payment_method,
                        'MonthlyCharges': monthly_charges,
                        'TotalCharges': total_charges
                    }
                    
                    # Make prediction
                    prediction = self.predict_churn(input_data)
                    
                    # Display results
                    st.success("Prediction completed!")
                    
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.metric("Churn Probability", f"{prediction['probability']:.2%}")
                    with col_pred2:
                        st.metric("Prediction", "Churn" if prediction['prediction'] == 1 else "No Churn")
                    
                    # Confidence visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prediction['probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk"},
                        gauge={'axis': {'range': [None, 100]},
                               'bar': {'color': "darkblue"},
                               'steps': [{'range': [0, 50], 'color': "lightgray"},
                                        {'range': [50, 75], 'color': "yellow"},
                                        {'range': [75, 100], 'color': "red"}],
                               'threshold': {'line': {'color': "red", 'width': 4},
                                           'thickness': 0.75, 'value': 75}}))
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance if available
                    if 'feature_importance' in prediction:
                        st.subheader("Feature Importance")
                        importance_df = pd.DataFrame({
                            'feature': list(prediction['feature_importance'].keys()),
                            'importance': list(prediction['feature_importance'].values())
                        }).sort_values('importance', ascending=False)
                        
                        fig_importance = px.bar(
                            importance_df.head(10),
                            x='importance',
                            y='feature',
                            orientation='h',
                            title="Top 10 Most Important Features"
                        )
                        st.plotly_chart(fig_importance, use_container_width=True)
    
    def predict_churn(self, input_data):
        """Predict churn for input data"""
        try:
            # Load production model - prefer xgboost
            models = self.registry.list_models("production")
            if not models:
                st.error("No production models available")
                return {'prediction': 0, 'probability': 0.0}
            
            # Find xgboost model, otherwise use latest
            xgboost_model = next((m for m in models if 'xgboost' in m['model_id'].lower()), None)
            latest_model = xgboost_model if xgboost_model else models[0]
            
            model = self.registry.load_model(latest_model["model_id"], "production")
            
            # Convert input data to features (this would need proper feature engineering)
            # For demonstration, using dummy conversion
            features = self._prepare_features(input_data)
            
            # Make prediction
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0][1] if hasattr(model, 'predict_proba') else float(prediction)
            
            # Get feature importance if available
            feature_importance = {}
            if hasattr(model, 'feature_importances_'):
                feature_names = self._get_feature_names()
                feature_importance = dict(zip(feature_names, model.feature_importances_))
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'feature_importance': feature_importance,
                'model_id': latest_model["model_id"]
            }
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return {'prediction': 0, 'probability': 0.0}
    
    def _prepare_features(self, input_data):
        """Prepare features matching training pipeline"""
        try:
            # Create DataFrame with all columns in correct order
            # Note: customerID is included as it was in the training pipeline
            customer_id_hash = hash('PRED-' + str(np.random.randint(1000, 9999))) % 10000
            
            df = pd.DataFrame([{
                'customerID': str(customer_id_hash), # Training pipeline uses .astype(str)
                'gender': input_data['gender'],
                'SeniorCitizen': input_data['SeniorCitizen'],
                'Partner': input_data['Partner'],
                'Dependents': input_data['Dependents'],
                'tenure': input_data['tenure'],
                'PhoneService': input_data['PhoneService'],
                'MultipleLines': input_data['MultipleLines'],
                'InternetService': input_data['InternetService'],
                'OnlineSecurity': input_data['OnlineSecurity'],
                'OnlineBackup': input_data['OnlineBackup'],
                'DeviceProtection': input_data['DeviceProtection'],
                'TechSupport': input_data['TechSupport'],
                'StreamingTV': input_data['StreamingTV'],
                'StreamingMovies': input_data['StreamingMovies'],
                'Contract': input_data['Contract'],
                'PaperlessBilling': input_data['PaperlessBilling'],
                'PaymentMethod': input_data['PaymentMethod'],
                'MonthlyCharges': input_data['MonthlyCharges'],
                'TotalCharges': input_data['TotalCharges']
            }])
            
            # Apply label encoding using saved encoders
            for col, encoder in self.data_loader.label_encoders.items():
                if col in df.columns:
                    try:
                        # Handle unknown categories by using the first class if not found
                        val = str(df[col].iloc[0])
                        if val in encoder.classes_:
                            df[col] = encoder.transform([val])[0]
                        else:
                            # Default to 0 if category not seen during training
                            df[col] = 0
                    except Exception as e:
                        logger.warning(f"Error encoding {col}: {e}")
                        df[col] = 0
            
            # Ensure all columns are numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Fallback for columns not in label_encoders
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        df[col] = 0
            
            return df.values
        except Exception as e:
            st.error(f"Feature preparation error: {e}")
            raise
    
    def _get_feature_names(self):
        """Get actual feature names from training schema"""
        return [
            "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", 
            "tenure", "PhoneService", "MultipleLines", "InternetService", 
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", 
            "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", 
            "PaymentMethod", "MonthlyCharges", "TotalCharges"
        ]
    
    def show_monitoring_page(self):
        """Performance monitoring interface"""
        st.header("📊 Performance Monitoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Model Performance")
            
            # Load performance data from MLflow or monitoring reports
            try:
                # Get recent performance metrics
                performance_data = self._load_performance_data()
                
                if performance_data:
                    # Display current metrics
                    st.metric("Accuracy", f"{performance_data.get('accuracy', 0):.3f}")
                    st.metric("Precision", f"{performance_data.get('precision', 0):.3f}")
                    st.metric("Recall", f"{performance_data.get('recall', 0):.3f}")
                    st.metric("F1 Score", f"{performance_data.get('f1_score', 0):.3f}")
                    
                    # Performance trends chart
                    st.subheader("Performance Trends")
                    trend_fig = self._create_performance_trend_chart(performance_data)
                    if trend_fig:
                        st.plotly_chart(trend_fig, use_container_width=True)
                else:
                    st.warning("No performance data available")
                    
            except Exception as e:
                st.error(f"Error loading performance data: {str(e)}")
        
        with col2:
            st.subheader("Data Drift Monitoring")
            
            # Data drift status
            drift_status = self._check_data_drift()
            
            if drift_status.get('drift_detected'):
                st.error("🚨 Data Drift Detected!")
                st.metric("Drift Score", f"{drift_status.get('drift_score', 0):.3f}")
                st.write(f"Drifted Columns: {len(drift_status.get('drifted_columns', []))}")
                
                if st.button("View Drift Report"):
                    st.write(drift_status.get('report_summary', {}))
            else:
                st.success("✅ No Data Drift")
                st.metric("Drift Score", f"{drift_status.get('drift_score', 0):.3f}")
            
            # Alert summary
            st.subheader("Alert Summary")
            alert_summary = self._get_alert_summary()
            
            col_alert1, col_alert2 = st.columns(2)
            with col_alert1:
                st.metric("Active Alerts", alert_summary.get('active_alerts', 0))
            with col_alert2:
                st.metric("Total Alerts", alert_summary.get('total_alerts', 0))
            
            # Alert breakdown
            st.write("**Alert Severity:**")
            for severity, count in alert_summary.get('severity_breakdown', {}).items():
                st.write(f"- {severity}: {count}")
    
    def _load_performance_data(self):
        """Load performance data from the latest production model"""
        try:
            # Get production models
            models = self.registry.list_models("production")
            if not models:
                # Try staging if no production model
                models = self.registry.list_models("staging")
            
            if not models:
                return None
            
            latest_model = models[0]
            metrics = latest_model.get('performance_metrics', {})
            
            if not metrics:
                # Handle different naming in metadata
                metrics = latest_model.get('metrics', {})
            
            if metrics:
                # Add trend data (dummy for now as we don't have historical performance file yet)
                metrics['trend_data'] = {
                    'timestamps': [datetime.now().strftime('%Y-%m-%d')],
                    'accuracy': [metrics.get('accuracy', 0)],
                    'precision': [metrics.get('precision', 0)],
                    'recall': [metrics.get('recall', 0)]
                }
                return metrics
            
            return None
            
        except Exception as e:
            logger.error(f"Error loading performance data: {e}")
            return None
    
    def _create_performance_trend_chart(self, performance_data):
        """Create performance trend chart"""
        if 'trend_data' not in performance_data:
            return None
        
        trend_data = performance_data['trend_data']
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=trend_data['timestamps'],
            y=trend_data['accuracy'],
            name='Accuracy',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['timestamps'],
            y=trend_data['precision'],
            name='Precision',
            line=dict(color='green')
        ))
        
        fig.add_trace(go.Scatter(
            x=trend_data['timestamps'],
            y=trend_data['recall'],
            name='Recall',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            title="Model Performance Trends",
            xaxis_title="Date",
            yaxis_title="Score",
            hovermode='x unified'
        )
        
        return fig
    
    def _check_data_drift(self):
        """Check for data drift from history"""
        try:
            drift_history_path = Path("reports/drift_reports/drift_history.json")
            if drift_history_path.exists():
                with open(drift_history_path, 'r') as f:
                    history = json.load(f)
                
                if history:
                    # Return the latest report
                    latest = history[-1]
                    return {
                        'drift_detected': latest.get('drift_detected', False),
                        'drift_score': latest.get('drift_score', 0),
                        'drifted_columns': latest.get('drifted_columns', []),
                        'report_summary': {'status': 'Drift analysis from ' + latest.get('timestamp', 'unknown')}
                    }
            
            return {
                'drift_detected': False,
                'drift_score': 0,
                'drifted_columns': [],
                'report_summary': {'status': 'No drift reports found'}
            }
        except Exception as e:
            logger.error(f"Error checking data drift: {e}")
            return {'drift_detected': False, 'drift_score': 0, 'drifted_columns': [], 'report_summary': {}}
    
    def _get_alert_summary(self):
        """Get alert summary"""
        # This would use the alert system
        # For now, return dummy data
        return {
            'active_alerts': 2,
            'total_alerts': 15,
            'severity_breakdown': {'HIGH': 1, 'MEDIUM': 1, 'LOW': 0, 'CRITICAL': 0}
        }
    
    def show_optuna_dashboard(self):
        """Optuna dashboard interface"""
        st.header("🔬 Optuna Dashboard")
        
        st.info("""
        Optuna dashboard provides interactive visualization of hyperparameter optimization studies.
        Access the full dashboard at: http://localhost:8080
        """)
        
        # Display study statistics if available
        try:
            studies = self.optuna_monitor.list_studies()
            
            if studies:
                st.subheader("Optimization Studies")
                
                for study in studies:
                    with st.expander(f"Study: {study['name']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Trials", study['n_trials'])
                            st.metric("Best Value", f"{study['best_value']:.4f}")
                        
                        with col2:
                            st.metric("Status", study['state'])
                            st.metric("Duration", f"{study['duration_hours']:.1f}h")
                
                # Show best parameters
                best_study = max(studies, key=lambda x: x['best_value'])
                st.subheader("Best Study Parameters")
                
                if best_study.get('best_params'):
                    best_params = best_study['best_params']
                    for param, value in best_params.items():
                        st.write(f"**{param}:** {value}")
                
                # Show optimization history chart
                if st.button("Show Optimization Progress"):
                    progress_fig = self.optuna_monitor.plot_optimization_history(best_study['name'])
                    if progress_fig:
                        st.plotly_chart(progress_fig, use_container_width=True)
            
            else:
                st.warning("No optimization studies found")
                
        except Exception as e:
            st.error(f"Error loading Optuna studies: {str(e)}")
        
        # Quick optimization controls
        st.subheader("Quick Optimization")
        
        if st.button("Run Quick Optimization", type="primary"):
            with st.spinner("Running optimization..."):
                try:
                    result = self.optuna_monitor.run_quick_optimization()
                    st.success(f"Optimization completed! Best score: {result['best_value']:.4f}")
                    st.json(result['best_params'])
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
    
    def show_data_analysis(self):
        """Data analysis interface"""
        st.header("📈 Data Analysis")
        
        # Data overview
        st.subheader("Dataset Overview")
        
        try:
            # Load sample data for analysis
            data_config = load_config("configs/data/local.yaml")
            data_path = data_config.get('train_data_path', 'data/processed/train.csv')
            
            if Path(data_path).exists():
                data = pd.read_csv(data_path)
                
                # Standardize column names
                if 'Churn' in data.columns:
                    data = data.rename(columns={'Churn': 'churn'})
                
                # Convert target to numeric if needed
                if 'churn' in data.columns and data['churn'].dtype == 'object':
                    data['churn'] = data['churn'].map({'Yes': 1, 'No': 0})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Samples", len(data))
                
                with col2:
                    st.metric("Features", len(data.columns) - 1)  # Exclude target
                
                with col3:
                    churn_rate = data['churn'].mean() if 'churn' in data.columns else 0
                    st.metric("Churn Rate", f"{churn_rate:.2%}")
                
                # Data distribution
                st.subheader("Data Distribution")
                
                # Feature distributions
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_feature = st.selectbox("Select feature to visualize", numeric_cols)
                    
                    if selected_feature:
                        fig_dist = px.histogram(data, x=selected_feature, 
                                              title=f"Distribution of {selected_feature}",
                                              color='churn' if 'churn' in data.columns else None)
                        st.plotly_chart(fig_dist, use_container_width=True)
                
                # Correlation heatmap
                if len(numeric_cols) > 1:
                    st.subheader("Correlation Heatmap")
                    
                    # Calculate correlation matrix
                    corr_matrix = data[numeric_cols].corr()
                    
                    fig_corr = px.imshow(corr_matrix,
                                       title="Feature Correlation Matrix",
                                       aspect="auto",
                                       color_continuous_scale='RdBu_r')
                    st.plotly_chart(fig_corr, use_container_width=True)
                
                # Target distribution
                if 'churn' in data.columns:
                    st.subheader("Target Distribution")
                    
                    churn_counts = data['churn'].value_counts()
                    fig_target = px.pie(values=churn_counts.values,
                                      names=churn_counts.index.map({0: 'No Churn', 1: 'Churn'}),
                                      title="Churn Distribution")
                    st.plotly_chart(fig_target, use_container_width=True)
            
            else:
                st.warning("Training data not found for analysis")
                
        except Exception as e:
            st.error(f"Error loading data for analysis: {str(e)}")
        
        # Feature importance analysis
        st.subheader("Feature Importance Analysis")
        
        if st.button("Analyze Feature Importance"):
            with st.spinner("Analyzing feature importance..."):
                try:
                    # This would use actual feature importance from models
                    # For now, show dummy analysis
                    importance_data = {
                        'feature': ['tenure', 'monthly_charges', 'total_charges', 'contract', 'internet_service'],
                        'importance': [0.35, 0.25, 0.20, 0.15, 0.05]
                    }
                    
                    fig_importance = px.bar(importance_data, x='importance', y='feature',
                                          orientation='h', title="Feature Importance Ranking")
                    st.plotly_chart(fig_importance, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Feature analysis failed: {str(e)}")

# Main application entry point
if __name__ == "__main__":
    app = ChurnPredictionApp()
    app.run()
