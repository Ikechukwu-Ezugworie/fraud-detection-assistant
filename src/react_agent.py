import logging
import os
from datetime import datetime

import joblib
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set up decision log file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
decision_log_handler = logging.FileHandler(os.path.join(LOG_DIR, "decisions.log"))
decision_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
decision_logger = logging.getLogger("decision_logger")
decision_logger.addHandler(decision_log_handler)
decision_logger.setLevel(logging.INFO)


class ReActAgent:
    def __init__(self, model_path="models/baseline_model.pkl", train_path="data/processed/train.csv"):
        """Initialize ReAct agent with trained model and historical data."""
        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded model from {model_path}")
        except FileNotFoundError:
            logger.error(f"Error: Model file {model_path} not found. Run model.py first.")
            exit(1)

        try:
            self.train_df = pd.read_csv(train_path)
            logger.info(f"Loaded training data from {train_path} for historical context.")
            # Compute historical statistics for reasoning
            self.amount_mean = self.train_df["Amount"].mean()
            self.amount_std = self.train_df["Amount"].std()
            self.time_mean = self.train_df["Time"].mean()
            self.time_std = self.train_df["Time"].std()
            # Store feature names for consistency
            self.feature_names = [col for col in self.train_df.columns if col != "Class"]
        except FileNotFoundError:
            logger.error(f"Error: Training data {train_path} not found. Run preprocess.py first.")
            exit(1)

        self.threshold = 0.0017  # Initial threshold based on dataset fraud ratio
        self.decision_log = []

    def observe(self, transaction):
        """Observe transaction context and return features with column names."""
        try:
            # Ensure transaction is a DataFrame with correct feature names
            if isinstance(transaction, pd.Series):
                transaction = transaction.to_frame().T
            features = transaction.drop(labels=["Class"] if "Class" in transaction.columns else [], axis=1)
            # Verify feature names match training data
            if list(features.columns) != self.feature_names:
                logger.error(f"Feature mismatch: Expected {self.feature_names}, got {list(features.columns)}")
                return None
            return features
        except Exception as e:
            logger.error(f"Error observing transaction: {str(e)}")
            return None

    def reason(self, features, anomaly_score):
        """Generate reasoning chain based on transaction features and historical patterns."""
        reasoning = []
        try:
            amount = features["Amount"].iloc[0]
            time = features["Time"].iloc[0]

            # Rule 1: Check if Amount deviates significantly from historical mean
            amount_z_score = (amount - self.amount_mean) / self.amount_std
            if abs(amount_z_score) > 3:
                reasoning.append(
                    f"Amount ({amount:.2f}) is {abs(amount_z_score):.2f} std devs from mean ({self.amount_mean:.2f})."
                )

            # Rule 2: Check if Time deviates significantly
            time_z_score = (time - self.time_mean) / self.time_std
            if abs(time_z_score) > 3:
                reasoning.append(
                    f"Time ({time:.2f}) is {abs(time_z_score):.2f} std devs from mean ({self.time_mean:.2f})."
                )

            # Rule 3: Check model anomaly score
            if anomaly_score == -1:
                reasoning.append("Isolation Forest flagged transaction as anomalous.")
            else:
                reasoning.append("Isolation Forest classified transaction as normal.")

            return reasoning if reasoning else ["No significant anomalies detected."]
        except Exception as e:
            logger.error(f"Error in reasoning: {str(e)}")
            return ["Error generating reasoning."]

    def decide(self, anomaly_score, reasoning):
        """Make decision (Flag/Approve) with adaptive thresholding."""
        try:
            # Decision: Flag if model predicts anomaly (-1)
            decision = 1 if anomaly_score == -1 else 0

            # Adaptive thresholding: Adjust based on reasoning severity
            if len([r for r in reasoning if "std devs" in r]) >= 2:
                decision = 1  # Override to flag if multiple severe deviations
                self.threshold = min(self.threshold * 1.1, 0.01)  # Increase threshold
            else:
                self.threshold = max(self.threshold * 0.95, 0.001)  # Decrease threshold

            return decision
        except Exception as e:
            logger.error(f"Error in decision: {str(e)}")
            return 0

    def act(self, features, decision, reasoning):
        """Log decision and return action."""
        action = "Flag" if decision == 1 else "Approve"
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "transaction": features.to_dict("records")[0],
            "decision": action,
            "reasoning": reasoning,
            "threshold": self.threshold,
        }
        self.decision_log.append(log_entry)
        decision_logger.info(f"Action: {action}, Reasoning: {reasoning}, Threshold: {self.threshold:.6f}")
        return action

    def process_transaction(self, transaction):
        """Process a single transaction through the ReAct pipeline."""
        features = self.observe(transaction)
        if features is None:
            return "Error processing transaction."

        # Get model prediction using DataFrame with feature names
        anomaly_score = self.model.predict(features)[0]
        reasoning = self.reason(features, anomaly_score)
        decision = self.decide(anomaly_score, reasoning)
        action = self.act(features, decision, reasoning)
        return action


def main():
    # Initialize agent
    agent = ReActAgent()

    # Load test data for batch processing
    try:
        test_df = pd.read_csv("data/processed/test.csv")
        logger.info(f"Loaded test data with {test_df.shape[0]} transactions.")
    except FileNotFoundError:
        logger.error("Error: 'test.csv' not found in 'data/processed/'. Run preprocess.py first.")
        exit(1)

    # Process a subset of transactions (e.g., first 100 for testing)
    subset = test_df.head(100)
    for idx, transaction in subset.iterrows():
        action = agent.process_transaction(transaction)
        logger.info(f"Transaction {idx}: {action}")

    # Save decision log to file
    try:
        pd.DataFrame(agent.decision_log).to_csv(os.path.join(LOG_DIR, "decision_log.csv"), index=False)
        logger.info("Decision log saved to logs/decision_log.csv.")
    except Exception as e:
        logger.error(f"Error saving decision log: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
