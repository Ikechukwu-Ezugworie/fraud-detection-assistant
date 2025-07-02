# Fraud Detection Assistant with Agentic AI

## Summary
This project builds a lightweight fraud detection assistant using the ReAct framework to identify suspicious credit card transactions.
It leverages the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), 
employing logic-based reasoning to flag anomalies without manual reconfiguration. 

## Setup
1. Clone repo: `git clone <repository-url>`
2. Create virtual environment: `python -m venv venv && source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Download Kaggle dataset from [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to `data/raw/creditcard.csv`.


## Usage
1. **Preprocess Data**:
   - Normalize `Amount` and `Time`, split into train/test sets:
     ```bash
     python src/preprocess.py
     ```
   - Outputs: `data/processed/train.csv`, `data/processed/test.csv`