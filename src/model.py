import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import joblib
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create models directory
os.makedirs('models', exist_ok=True)


def load_data(train_path, test_path):
    """Load preprocessed train and test datasets."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        logger.info("Train and test datasets loaded successfully.")
        logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df
    except FileNotFoundError:
        logger.error("Error: 'train.csv' or 'test.csv' not found in 'data/processed/'. Run preprocess.py first.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        exit(1)


def train_isolation_forest(train_df):
    """Train Isolation Forest on normal transactions."""
    try:
        # Filter normal transactions (Class == 0) for training
        normal_train = train_df[train_df['Class'] == 0]
        X_train = normal_train.drop(columns=['Class'])
        logger.info(f"Training Isolation Forest on {X_train.shape[0]} normal transactions.")

        # Initialize and train Isolation Forest
        model = IsolationForest(
            contamination=0.0017,  # Approximate fraud ratio in dataset
            random_state=42,
            n_estimators=100,
            max_samples='auto',
            n_jobs=-1  # Use all available cores
        )
        model.fit(X_train)
        logger.info("Isolation Forest training completed.")
        return model
    except Exception as e:
        logger.error(f"Error training Isolation Forest: {str(e)}")
        exit(1)


def evaluate_model(model, test_df):
    """Evaluate model on test set using precision, recall, and F1-score."""
    try:
        X_test = test_df.drop(columns=['Class'])
        y_test = test_df['Class']

        # Predict anomalies (-1 for anomalies, 1 for normal)
        y_pred = model.predict(X_test)
        # Convert to binary: 1 (anomaly/fraud), 0 (normal)
        y_pred_binary = np.where(y_pred == -1, 1, 0)

        # Calculate metrics
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)

        logger.info("\nEvaluation Metrics:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred_binary)}")

        return precision, recall, f1
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        exit(1)


def save_model(model, path='models/baseline_model.pkl'):
    """Save trained model to disk."""
    try:
        joblib.dump(model, path)
        logger.info(f"Model saved to {path}.")
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        exit(1)


def main():
    # Load preprocessed data
    train_path = 'data/processed/train.csv'
    test_path = 'data/processed/test.csv'
    train_df, test_df = load_data(train_path, test_path)

    # Train Isolation Forest
    model = train_isolation_forest(train_df)

    # Evaluate model
    precision, recall, f1 = evaluate_model(model, test_df)

    # Save model
    save_model(model)

    # Verify saved model
    if os.path.exists('models/baseline_model.pkl'):
        logger.info("Verification: Model successfully saved.")
    else:
        logger.error("Verification failed: Model file not found.")


if __name__ == "__main__":
    main()