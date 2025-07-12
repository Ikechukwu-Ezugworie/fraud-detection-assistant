# Fraud Detection Assistant with Agentic AI

This project presents a fraud detection assistant powered by Agentic AI, designed to autonomously analyze and reason about credit card transactions. Leveraging the ReAct (Reasoning and Acting) framework, the assistant not only detects suspicious activity but also explains its decisions through interpretable reasoning steps. Built on the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), it showcases how Agentic AI systems can combine data-driven insights with dynamic decision-making to improve the accuracy and transparency of fraud detection.

## Getting Started

### Prerequisites

- [Python](https://www.python.org/downloads/) >= 3.12
- [uv](https://github.com/astral-sh/uv#installation)

### Setup

1.  **Clone the repository:**

    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Install dependencies using uv:**

    ```bash
     uv sync
    ```

3.  Download [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) to `data/raw/creditcard.csv`.

### Usage

1. **Preprocess Data**:
   Normalize `Amount` and `Time`, split into train/test sets:

   ```bash
   uv run prep
   ```

   > Outputs: `data/processed/train.csv`, `data/processed/test.csv`

2. **Train Baseline Model**:
   Train Isolation Forest on normal transactions and evaluate:

   ```bash
   uv run train
   ```

   > Outputs: `models/baseline_model.pkl`

3. **Test ReAct Agent**:
   Process 100 transactions, generate reasoning, and log decisions:

   ```bash
     uv run agent
   ```

   > Outputs: `logs/decisions.log` , `logs/decision_log.csv`
