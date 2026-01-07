import mlflow
from src.models.model_trainer import ModelTrainer
from src.utils.mlflow_utils import start_mlflow_run, log_params, log_metrics, log_model
from src.data.data_loader import DataLoader
import numpy as np

def run_training(max_iter):
    data_loader = DataLoader(config_path="configs/data/local.yaml")
    train_df = data_loader.load_processed_data("train")
    X_train, y_train = data_loader.get_feature_target_split(train_df)

    trainer = ModelTrainer()
    trainer.model.max_iter = max_iter

    with start_mlflow_run(run_name=f"Churn_Logistic_Regression_Training_max_iter_{max_iter}"):
        log_params({"model_type": "LogisticRegression", "max_iter": max_iter})
        model = trainer.train(X_train, y_train)

        val_df = data_loader.load_processed_data("validation")
        X_val, y_val = data_loader.get_feature_target_split(val_df)
        accuracy = model.score(X_val, y_val)
        log_metrics({"validation_accuracy": accuracy})

        input_example = np.array(X_train[:5])
        log_model(model, input_example=input_example)

def main():
    max_iters = [100, 500, 1000, 1500]
    for max_iter in max_iters:
        run_training(max_iter)

if __name__ == "__main__":
    main()
