import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Create processed directory
os.makedirs("data/processed", exist_ok=True)


def load_and_preprocess_chunks(file_path, chunksize=10000):
    """Load dataset in chunks and preprocess to optimize memory usage."""
    chunks = []
    scaler = StandardScaler()

    # Initialize variables for Time normalization
    max_time = None

    try:
        # First pass: get max_time for normalization and check data integrity
        logger.info("Scanning dataset for max Time value...")
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            if max_time is None or chunk["Time"].max() > max_time:
                max_time = chunk["Time"].max()
            if chunk.isnull().any().any():
                logger.warning("Missing values detected in chunk. Consider data cleaning.")

        # Second pass: preprocess chunks
        logger.info("Processing dataset in chunks...")
        for chunk in pd.read_csv(file_path, chunksize=chunksize):
            # Normalize Amount
            chunk["Amount"] = scaler.fit_transform(chunk[["Amount"]])
            # Normalize Time
            chunk["Time"] = chunk["Time"] / max_time
            chunks.append(chunk)

        # Concatenate chunks
        df = pd.concat(chunks, axis=0, ignore_index=True)
        logger.info("Dataset loaded and preprocessed successfully.")
        return df, scaler
    except FileNotFoundError:
        logger.error(
            "Error: 'creditcard.csv' not found in 'data/raw/'. "
            "Please download from Kaggle using this link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud "
            "and place it in 'data/raw/'."
        )
        exit(1)
    except Exception as e:
        logger.error(f"Error during loading/preprocessing: {str(e)}")
        exit(1)


def main():
    # Load and preprocess dataset
    file_path = "data/raw/creditcard.csv"
    df, scaler = load_and_preprocess_chunks(file_path, chunksize=10000)

    # Inspect dataset
    logger.info("Dataset Info:")
    logger.info(f"\n{df.info()}")
    logger.info("\nMissing Values:")
    logger.info(f"\n{df.isnull().sum()}")
    logger.info("\nPost-Normalization Stats:")
    logger.info(f"\n{df[['Amount', 'Time']].describe()}")

    # Verify class distribution
    fraud_ratio = df["Class"].mean()
    logger.info(f"Fraud ratio: {fraud_ratio:.4f} (should be ~0.0017)")

    # Split into train/test sets (80/20, stratified)
    try:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["Class"])
        logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
    except ValueError as e:
        logger.error(f"Error during train/test split: {str(e)}")
        exit(1)

    # Save processed datasets
    try:
        train_df.to_csv("data/processed/train.csv", index=False)
        test_df.to_csv("data/processed/test.csv", index=False)
        logger.info("Train and test sets saved to 'data/processed/'.")
    except Exception as e:
        logger.error(f"Error saving files: {str(e)}")
        exit(1)

    # Verify saved files
    if os.path.exists("data/processed/train.csv") and os.path.exists("data/processed/test.csv"):
        logger.info("Verification: Processed files successfully saved.")
    else:
        logger.error("Verification failed: Processed files not found.")


if __name__ == "__main__":
    main()
