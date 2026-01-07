import joblib
import os
from src.models.model_trainer import ModelTrainer  # Import the ModelTrainer class

# Create an instance of ModelTrainer
trainer = ModelTrainer()

# Define the model and preprocessor paths
model_path = "models/production/model.pkl"
preprocessor_path = "models/production/preprocessor.pkl"

# Save the model
joblib.dump(trainer.model, model_path)
print(f"Model saved to {model_path}")

# Save the preprocessor if it exists
if hasattr(trainer, 'preprocessor') and trainer.preprocessor is not None:
    joblib.dump(trainer.preprocessor, preprocessor_path)
    print(f"Preprocessor saved to {preprocessor_path}")
else:
    print("No preprocessor to save.")
