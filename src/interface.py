import logging

import pandas as pd
import streamlit as st

from src.react_agent import ReActAgent

logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Fraud Detection Assistant", layout="wide")


# Initialize ReAct agent
@st.cache_resource
def load_agent():
    try:
        agent = ReActAgent()
        logger.info("ReAct agent loaded successfully.")
        return agent
    except Exception as e:
        logger.error(f"Error loading ReAct agent: {str(e)}")
        st.error("Failed to load ReAct agent. Check logs for details.")
        return None


# Load test data for dropdown
@st.cache_data
def load_test_data():
    try:
        test_df = pd.read_csv("data/processed/test.csv")
        # Get 3 non-fraudulent and 2 fraudulent samples
        approve_samples = test_df[test_df["Class"] == 0].head(3)
        fraud_samples = test_df[test_df["Class"] == 1].head(2)
        samples = pd.concat([approve_samples, fraud_samples])
        return samples
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return pd.DataFrame()


# Main app
def main():
    st.title("🛡️ Fraud Detection Assistant Agent")
    st.markdown("An interactive demonstration of a ReAct-based AI agent for fraud detection.")

    agent = load_agent()
    if agent is None:
        return

    test_data = load_test_data()
    if test_data.empty:
        st.error("Could not load test data. Please ensure 'data/processed/test.csv' exists.")
        return

    sample_options = {
        f"Sample {i + 1} ({'Approve' if row.Class == 0 else 'Fraudulent'})": row
        for i, row in enumerate(test_data.itertuples(index=False))
    }

    st.header("Test a New Transaction")
    st.markdown("Select a pre-defined test case to see the agent's reasoning process.")

    selected_sample_key = st.selectbox(
        "Select a Test Case",
        options=list(sample_options.keys()),
    )
    selected_sample = sample_options[selected_sample_key]

    # Use a form to group inputs and the submission button
    with st.form("transaction_form"):
        col1, col2 = st.columns([1, 1], gap="large")

        with col1:
            st.subheader("Transaction Details")
            amount_input = st.number_input(
                "Transaction Amount",
                value=float(getattr(selected_sample, "Amount", 0.0)),
                disabled=True,
            )
            time_input = st.number_input(
                "Time of Day",
                value=float(getattr(selected_sample, "Time", 0.0)),
                disabled=True,
            )
            expected_decision = "Approve" if selected_sample.Class == 0 else "Flag"
            st.info(f"**Expected Decision**: {expected_decision}")

        with col2:
            st.subheader("Anonymized Features (V1-V28)")
            with st.expander("View V1-V28 Features", expanded=True):
                v_inputs = {}
                for i in range(1, 29):
                    feature_name = f"V{i}"
                    v_inputs[feature_name] = st.number_input(
                        feature_name,
                        value=float(getattr(selected_sample, feature_name, 0.0)),
                        disabled=True,
                        key=f"v{i}",
                    )

        submitted = st.form_submit_button("Analyze Transaction", type="primary", use_container_width=True)

    if submitted:
        if agent:
            st.markdown("---")
            st.subheader("Agent Analysis")
            try:
                # Reconstruct the transaction data for the agent
                transaction_data = {
                    "Time": selected_sample.Time,
                    "Amount": selected_sample.Amount,
                    **{f"V{i}": getattr(selected_sample, f"V{i}") for i in range(1, 29)},
                }
                transaction = pd.DataFrame([transaction_data])

                with st.spinner("Agent is thinking..."):
                    # The agent's process_transaction method doesn't require a user_id
                    reasoning_chain = agent.process_transaction(transaction.iloc[0])

                st.subheader("Result")
                st.info(f"**Expected Decision**: {expected_decision}")
                st.markdown("**Agent's Final Answer**:")
                if reasoning_chain:
                    reasoning_chain.pop(0)
                    reasoning_chain.pop(0)
                    for message in reasoning_chain:
                        st.code(f"{message.type}: {message.content}", language="text")
                else:
                    st.error("No reasoning chain found.")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
        else:
            st.error("ReAct agent not initialized.")


if __name__ == "__main__":
    main()
