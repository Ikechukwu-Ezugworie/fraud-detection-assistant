import logging

import joblib
import pandas as pd

# Set up logging
logger = logging.getLogger(__name__)

# Load the pre-trained Isolation Forest model
try:
    model = joblib.load("models/baseline_model.pkl")
    logger.info("Isolation Forest model loaded for tools.")
except FileNotFoundError:
    logger.error("Model file 'models/baseline_model.pkl' not found. Make sure to run model.py first.")
    model = None
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None


def get_anomaly_score(transaction: pd.DataFrame) -> str:
    """
    Calculates the anomaly score for a given transaction using the pre-trained Isolation Forest model.
    The model returns -1 for anomalies (potential fraud) and 1 for normal transactions.
    We convert this to a more descriptive string.
    """
    if model is None:
        return "Error: Anomaly detection model is not available."
    try:
        # The model expects a 2D array, so we pass the DataFrame directly
        score = model.predict(transaction)[0]
        if score == -1:
            return "High Anomaly Score (Potential Fraud)"
        else:
            return "Normal Score"
    except Exception as e:
        logger.error(f"Error getting anomaly score: {str(e)}")
        return f"Error: Could not calculate anomaly score. Details: {str(e)}"


import json

# Load user history data
try:
    with open("data/user_transaction_history.json", "r") as f:
        user_db = json.load(f)
    logger.info("User transaction history loaded for tools.")
except FileNotFoundError:
    logger.error("User history file not found. The 'get_user_history' tool will not work.")
    user_db = {}
except json.JSONDecodeError:
    logger.error("Error decoding user_transaction_history.json. The file might be corrupted.")
    user_db = {}

def get_user_history(user_id: str, transaction_amount: float) -> str:
    """
    Analyzes the user's transaction history by looking up the user_id in the loaded JSON file.
    It compares the current transaction amount against the user's historical average and standard deviation.
    """
    user_id_str = str(user_id)
    if not user_db or user_id_str not in user_db:
        return "User has no significant transaction history (New or Unknown User)."

    user_data = user_db[user_id_str]
    avg_amount = user_data.get("avg_transaction_amount", 0)
    std_dev = user_data.get("std_dev_amount", 0)
    profile = user_data.get("profile", "N/A")

    if avg_amount == 0 and std_dev == 0:
        return f"User profile '{profile}' has no significant transaction history."

    if transaction_amount > avg_amount + (3 * std_dev):
        return f"Transaction amount is significantly higher than the user's average of ${avg_amount:.2f} (Profile: {profile})."
    elif transaction_amount < avg_amount - (3 * std_dev) and transaction_amount > 0:
        return f"Transaction amount is significantly lower than the user's average of ${avg_amount:.2f} (Profile: {profile})."
    else:
        return f"Transaction amount is consistent with the user's history (Average: ${avg_amount:.2f}, Profile: {profile})."


def analyze_contextual_patterns(transaction_time: float, transaction_amount: float) -> str:
    """
    Analyzes contextual patterns of the transaction, like time of day or amount.
    (Placeholder) This tool uses simple heuristics to detect common fraud patterns.
    """
    # Time is normalized between 0 and 1. Let's assume 0.9-1.0 and 0.0-0.1 are "late night".
    is_late_night = 0.9 <= transaction_time <= 1.0 or 0.0 <= transaction_time <= 0.1
    is_round_number = transaction_amount % 100 == 0 and transaction_amount > 0

    patterns = []
    if is_late_night:
        patterns.append("Transaction occurred at an unusual time (late night).")
    if is_round_number:
        patterns.append("Transaction is a large, round number, which can be a fraud indicator.")

    if not patterns:
        return "No suspicious contextual patterns detected."

    return " ".join(patterns)
