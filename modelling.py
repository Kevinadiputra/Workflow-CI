import os
import argparse
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description="Train Heart Disease Classifier")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=5, help="Maximum tree depth")
    args = parser.parse_args()
    
    # Set experiment
    mlflow.set_experiment("Heart_Disease_Workflow_CI")
    
    # Load dataset
    # By default, it will load from dataset/heart_disease_preprocessed.csv
    dataset_path = os.path.join("dataset", "heart_disease_preprocessed.csv")
    if not os.path.exists(dataset_path):
        # Check parent folder or workspace fallback if run outside the target directory
        fallback_path = os.path.join("Workflow-CI", "dataset", "heart_disease_preprocessed.csv")
        if os.path.exists(fallback_path):
            dataset_path = fallback_path
        else:
            raise FileNotFoundError(f"Preprocessed dataset not found at {dataset_path} or {fallback_path}")
        
    print(f"Loading preprocessed dataset from {dataset_path}...")
    df = pd.read_csv(dataset_path)
    
    X = df.drop(columns=['target'])
    y = df['target']
    
    # MLflow Training Run
    with mlflow.start_run() as run:
        print(f"Training RandomForestClassifier(n_estimators={args.n_estimators}, max_depth={args.max_depth})...")
        
        # Log parameters
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        
        # Initialize and train
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X, y)
        
        # Evaluate
        preds = model.predict(X)
        acc = accuracy_score(y, preds)
        
        # Log metrics
        mlflow.log_metric("train_accuracy", acc)
        print(f"Logged Training Accuracy: {acc:.4f}")
        
        # Log model artifact to MLflow server
        mlflow.sklearn.log_model(model, "model")
        
        # Also save the model locally for Docker packaging
        local_model_path = "model"
        if os.path.exists(local_model_path):
            shutil.rmtree(local_model_path)
        mlflow.sklearn.save_model(model, local_model_path)
        print(f"Saved model locally to directory: '{local_model_path}' for Docker serving")
        
        print("Model training successfully tracked via MLflow Project!")

if __name__ == "__main__":
    main()
