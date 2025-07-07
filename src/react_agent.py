import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import time
from huggingface_hub import InferenceClient
import huggingface_hub

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set up decision log file
LOG_DIR = 'logs'
os.makedirs(LOG_DIR, exist_ok=True)
decision_log_handler = logging.FileHandler(os.path.join(LOG_DIR, 'decisions.log'))
decision_log_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
decision_logger = logging.getLogger('decision_logger')
decision_logger.addHandler(decision_log_handler)
decision_logger.setLevel(logging.INFO)


class ReActAgent:
    def __init__(self, model_path='models/baseline_model.pkl', train_path='data/processed/train.csv', hf_token=None):
        """Initialize ReAct agent with trained model, historical data, and Qwen 2.5 7B Inference API."""
        # Log script version and environment
        logger.info(f"Script version: 2025-07-07 v2, huggingface_hub version: {huggingface_hub.__version__}")

        try:
            self.model = joblib.load(model_path)
            logger.info(f"Loaded Isolation Forest model from {model_path}")
        except FileNotFoundError:
            logger.error(f"Error: Model file {model_path} not found. Run model.py first.")
            exit(1)

        try:
            self.train_df = pd.read_csv(train_path)
            logger.info(f"Loaded training data from {train_path} for historical context.")
            self.amount_mean = self.train_df['Amount'].mean()
            self.amount_std = self.train_df['Amount'].std()
            self.time_mean = self.train_df['Time'].mean()
            self.time_std = self.train_df['Time'].std()
            self.feature_names = [col for col in self.train_df.columns if col != 'Class']
        except FileNotFoundError:
            logger.error(f"Error: Training data {train_path} not found. Run preprocess.py first.")
            exit(1)

        # Initialize Inference API client for Qwen 2.5 7B
        try:
            # self.client = InferenceClient(model="Qwen/Qwen2.5-7B-Instruct", token=hf_token)
            self.client = InferenceClient(
                model="Qwen/Qwen2.5-VL-7B-Instruct",  # confirmed inference-enabled
                token=hf_token,
                provider="hf-inference"
            )
            logger.info("Initialized Inference API client for Qwen 2.5 7B.")
        except Exception as e:
            logger.warning(
                f"Failed to initialize Inference API client: {str(e)}. Falling back to rule-based reasoning.")
            self.client = None

        self.threshold = 0.0017  # Initial threshold based on dataset fraud ratio
        self.decision_log = []

    def observe(self, transaction):
        """Observe transaction context and return features with column names."""
        try:
            if isinstance(transaction, pd.Series):
                transaction = transaction.to_frame().T
            features = transaction.drop(labels=['Class'] if 'Class' in transaction.columns else [], axis=1)
            if list(features.columns) != self.feature_names:
                logger.error(f"Feature mismatch: Expected {self.feature_names}, got {list(features.columns)}")
                return None
            return features
        except Exception as e:
            logger.error(f"Error observing transaction: {str(e)}")
            return None

    def reason(self, features, anomaly_score):
        """Generate reasoning chain using Qwen 2.5 7B Inference API or rule-based fallback."""
        try:
            amount = features['Amount'].iloc[0]
            txn_time = features['Time'].iloc[0]  # Avoid shadowing 'time' module

            if self.client:
                # Prepare prompt for Qwen 2.5
                prompt = (
                    f"A credit card transaction has normalized Amount={amount:.2f} "
                    f"(historical mean={self.amount_mean:.2f}, std={self.amount_std:.2f}) and "
                    f"Time={txn_time:.2f} (historical mean={self.time_mean:.2f}, std={self.time_std:.2f}). "
                    f"The Isolation Forest model predicts: {'anomaly' if anomaly_score == -1 else 'normal'}. "
                    f"Provide a concise reasoning chain (1-2 sentences) to determine if this transaction is likely fraudulent."
                )
                # Retry logic for API rate limits
                for attempt in range(3):
                    try:
                        response = self.client.text_generation(prompt, max_new_tokens=100, temperature=0.7)
                        reasoning = [response.strip()]
                        if not reasoning[0]:
                            raise ValueError("Empty response from API")
                        return reasoning
                    except Exception as api_error:
                        logger.warning(f"API attempt {attempt + 1} failed: {str(api_error)}")
                        time.sleep(2 ** attempt)  # Exponential backoff
                logger.error("All API attempts failed. Falling back to rule-based reasoning.")
            # Rule-based reasoning fallback
            reasoning = []
            amount_z_score = (amount - self.amount_mean) / self.amount_std
            if abs(amount_z_score) > 3:
                reasoning.append(
                    f"Amount ({amount:.2f}) is {abs(amount_z_score):.2f} std devs from mean ({self.amount_mean:.2f}).")
            time_z_score = (txn_time - self.time_mean) / self.time_std
            if abs(time_z_score) > 3:
                reasoning.append(
                    f"Time ({txn_time:.2f}) is {abs(time_z_score):.2f} std devs from mean ({self.time_mean:.2f}).")
            if anomaly_score == -1:
                reasoning.append("Isolation Forest flagged transaction as anomalous.")
            else:
                reasoning.append("Isolation Forest classified transaction as normal.")
            reasoning = reasoning if reasoning else ["No significant anomalies detected."]
            return reasoning
        except Exception as e:
            logger.error(f"Error in reasoning: {str(e)}")
            return ["Error generating reasoning."]

    def decide(self, anomaly_score, reasoning):
        """Make decision (Flag/Approve) with adaptive thresholding."""
        try:
            decision = 1 if anomaly_score == -1 else 0
            if len([r for r in reasoning if "std devs" in r or "fraud" in r.lower()]) >= 2:
                decision = 1
                self.threshold = min(self.threshold * 1.1, 0.01)
            else:
                self.threshold = max(self.threshold * 0.95, 0.001)
            return decision
        except Exception as e:
            logger.error(f"Error in decision: {str(e)}")
            return 0

    def act(self, features, decision, reasoning):
        """Log decision and return action."""
        action = "Flag" if decision == 1 else "Approve"
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'transaction': features.to_dict('records')[0],
            'decision': action,
            'reasoning': reasoning,
            'threshold': self.threshold
        }
        self.decision_log.append(log_entry)
        decision_logger.info(f"Action: {action}, Reasoning: {reasoning}, Threshold: {self.threshold:.6f}")
        return action

    def process_transaction(self, transaction):
        """Process a single transaction through the ReAct pipeline."""
        features = self.observe(transaction)
        if features is None:
            return "Error processing transaction."
        anomaly_score = self.model.predict(features)[0]
        reasoning = self.reason(features, anomaly_score)
        decision = self.decide(anomaly_score, reasoning)
        action = self.act(features, decision, reasoning)
        return action


def main():
    # Hugging Face token (optional for Qwen 2.5, set to None)
    hf_token = os.getenv("HF_TOKEN")  # Not required for Qwen 2.5

    # Initialize agent
    agent = ReActAgent(hf_token=hf_token)

    # Load test data for batch processing
    try:
        test_df = pd.read_csv('data/processed/test.csv')
        logger.info(f"Loaded test data with {test_df.shape[0]} transactions.")
    except FileNotFoundError:
        logger.error("Error: 'test.csv' not found in 'data/processed/'. Run preprocess.py first.")
        exit(1)

    # Process a small subset of transactions (5 for API efficiency)
    subset = test_df.head(5)
    for idx, transaction in subset.iterrows():
        action = agent.process_transaction(transaction)
        logger.info(f"Transaction {idx}: {action}")

    # Save decision log
    try:
        pd.DataFrame(agent.decision_log).to_csv(os.path.join(LOG_DIR, 'decision_log.csv'), index=False)
        logger.info("Decision log saved to logs/decision_log.csv.")
    except Exception as e:
        logger.error(f"Error saving decision log: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()