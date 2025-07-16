import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.react_agent import ReActAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(page_title="Fraud Detection Assistant", layout="wide")


# Initialize ReAct agent
@st.cache_resource
def load_agent():
    try:
        hf_token = os.getenv("HF_TOKEN")
        hf_model = os.getenv("HF_MODEL", "Gensyn/Qwen2.5-0.5B-Instruct")
        agent = ReActAgent(hf_token=hf_token, hf_model=hf_model)
        logger.info("ReAct agent loaded successfully.")
        return agent
    except Exception as e:
        logger.error(f"Error loading ReAct agent: {str(e)}")
        st.error("Failed to load ReAct agent. Check logs for details.")
        return None


# Load user data for dropdown
@st.cache_data
def load_user_profiles():
    try:
        with open("data/user_transaction_history.json", "r") as f:
            users = json.load(f)
        return {user_id: f"{data['name']} ({data['profile']})" for user_id, data in users.items()}
    except Exception as e:
        logger.error(f"Error loading user profiles: {e}")
        return {}


# Main app
def main():
    st.title("🛡️ Fraud Detection Assistant Agent")
    st.markdown("An interactive demonstration of a ReAct-based AI agent for fraud detection.")

    agent = load_agent()
    if agent is None:
        return

    # Main container for the interactive demo
    st.header("Test a New Transaction")
    st.markdown("Select a user profile and enter transaction details to see the agent's reasoning process.")

    user_profiles = load_user_profiles()
    if not user_profiles:
        st.error(
            "Could not load user profiles. Please ensure 'data/user_transaction_history.json' exists and is valid."
        )
        return

    with st.form("transaction_form"):
        col1, col2 = st.columns([1, 1], gap="large")
        with col1:
            st.subheader("Transaction Details")
            selected_user_id = st.selectbox(
                "Select a User Profile",
                options=list(user_profiles.keys()),
                format_func=lambda x: user_profiles.get(x, "Unknown"),
            )
            amount_input = st.number_input(
                "Transaction Amount (e.g., 125.50)",
                value=st.session_state.get("amount_input", 100.0),
                step=10.0,
            )
            time_input = st.slider(
                "Time of Day (0.0 = Midnight, 1.0 = 11:59 PM)",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.get("time_input", 0.5),
                step=0.01,
            )

        with col2:
            st.subheader("Anonymized Features (V1-V28)")
            with st.expander("Expand to modify V1-V28 features"):
                v_inputs = {
                    f"V{i}": st.number_input(
                        f"V{i}",
                        value=st.session_state.get("v_inputs", {}).get(f"V{i}", 0.0),
                        step=0.1,
                        key=f"v{i}",
                    )
                    for i in range(1, 29)
                }

        submitted = st.form_submit_button("Analyze Transaction", type="primary", use_container_width=True)

    if st.button("Randomize Features", use_container_width=True):
        st.session_state["amount_input"] = np.random.uniform(1, 5000)
        st.session_state["time_input"] = np.random.uniform(0, 1)
        st.session_state["v_inputs"] = {f"V{i}": np.random.uniform(-5, 5) for i in range(1, 29)}
        st.rerun()

    if submitted:
        if agent and selected_user_id:
            st.markdown("---")
            st.subheader("Agent Analysis")
            try:
                # The model expects normalized data, but for a better UX, we let users input raw amounts.
                # We'll use a placeholder normalization for the demo. A real system would use a fitted scaler.
                normalized_amount = (amount_input - 122) / 256  # Placeholder based on original dataset stats

                transaction_data = {
                    "Time": time_input,
                    "Amount": normalized_amount,
                    **v_inputs,
                }
                transaction = pd.DataFrame([transaction_data])

                with st.spinner("Agent is thinking..."):
                    action, reasoning_chain = agent.process_transaction(transaction.iloc[0], user_id=selected_user_id)

                st.subheader("Result")
                if action == "Flag":
                    st.error(f"**Decision**: {action}", icon="🚨")
                else:
                    st.success(f"**Decision**: {action}", icon="✅")

                st.markdown("**Agent's Reasoning Chain**:")
                st.code("\n".join(reasoning_chain), language="text")

            except Exception as e:
                st.error(f"An error occurred during processing: {e}")
        elif not agent:
            st.error("ReAct agent not initialized.")
        else:
            st.error("Please select a user.")


if __name__ == "__main__":
    main()
