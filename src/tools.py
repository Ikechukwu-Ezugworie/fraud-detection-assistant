import logging
import os

import joblib
import pandas as pd
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

load_dotenv()

model = None
try:
    model = joblib.load("models/baseline_model.pkl")
    logger.info("Isolation Forest model loaded for tools.")
except Exception as e:
    logger.error(f"Failed to load Isolation Forest model: {e}")


hf_token = os.getenv("HF_TOKEN")
hf_inference_model = os.getenv("HF_INFERENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")
try:
    client = InferenceClient(
        model=hf_inference_model,
        token=hf_token,
    )
    logger.info(f"Initialized Inference API client for {hf_inference_model}.")
except Exception as e:
    logger.error(f"Failed to initialize Inference API client: {str(e)}.")
    client = None


train_path = "data/processed/train.csv"
try:
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded training data from {train_path} for historical context.")
    amount_mean = train_df["Amount"].mean()
    amount_std = train_df["Amount"].std()
    time_mean = train_df["Time"].mean()
    time_std = train_df["Time"].std()
except FileNotFoundError:
    logger.error(f"Error: Training data {train_path} not found. Run preprocess.py first.")
    exit(1)


@tool
def get_anomaly_score(features: dict) -> int:
    """
    Calculates an anomaly score for a transaction using a pre-trained Isolation Forest model.

    Args:
        features (dict): A dictionary containing the transaction features,
                         maintaining the order: 'Time', 'V1', 'V2', ... 'V28', 'Amount'.

    Returns:
        int: The anomaly score. Returns -1 for an anomaly (potential fraud) and 1 for a normal transaction.
             Returns 0 if the model is not available or an error occurs.
    """

    if model is None:
        logger.error("Anomaly detection model is not available.")
        return 0
    try:
        # The agent might pass the features nested inside another dictionary.
        if "features" in features:
            features = features["features"]

        ordered_features = {}
        ordered_features["Time"] = features.get("Time")
        for i in range(1, 29):
            ordered_features[f"V{i}"] = features.get(f"V{i}")
        ordered_features["Amount"] = features.get("Amount")

        transaction_df = pd.DataFrame([ordered_features])
        score = model.predict(transaction_df)[0]
        return int(score)
    except Exception as e:
        logger.error(f"Error in get_anomaly_score: {e}")
        return 0


@tool
def get_decision_with_reasoning(amount: float, time: float, anomaly_score: int) -> str:
    """
    Generates a natural language explanation for a transaction's fraud risk using a Hugging Face model.

    Args:
        amount (float): The normalized transaction amount.
        time (float): The normalized transaction time.
        anomaly_score (int): The anomaly score from the Isolation Forest model (-1 for anomaly, 1 for normal).

    Returns:
        str: A concise, human-readable explanation of the fraud risk.
    """
    if client:
        prompt = (
            f"A credit card transaction has normalized Amount={amount:.2f} "
            f"(historical mean={amount_mean:.2f}, std={amount_std:.2f}) and "
            f"Time={time:.2f} (historical mean={time_mean:.2f}, std={time_std:.2f}). "
            f"The Isolation Forest model predicts: {'anomaly' if anomaly_score == -1 else 'normal'}. "
            f"Provide a concise reasoning chain (1-2 sentences) to determine if this transaction is likely fraudulent."
        )

        try:
            response = (
                client.chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.5,
                )
                .choices[0]
                .message.content
            )
            if not response:
                raise ValueError("API returned an empty response.")

            return response
        except Exception as e:
            logger.error(f"Error calling HF API: {e}")
            return f"Error: API call failed. Details: {e}"

    return "Error: Inference API client not initialized."


all_tools = [get_anomaly_score, get_decision_with_reasoning]
