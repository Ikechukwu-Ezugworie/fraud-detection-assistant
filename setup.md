
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

3. **Copy and fill the .env variables
    ```bash
    cp .env.example .env
    ```

4. **Run Streamlit App**:

   ```bash
   streamlit run src/interface.py
   ```