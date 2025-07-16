import logging
import os
import time
from datetime import datetime

import pandas as pd
from huggingface_hub import InferenceClient
from huggingface_hub.errors import HfHubHTTPError

from src import tools

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
    def __init__(self, hf_token=None, hf_model="Qwen/Qwen2.5-7B-Instruct"):
        """Initialize the ReAct agent with a Hugging Face model and a set of tools."""
        self.client = self._initialize_hf_client(hf_model, hf_token)
        self.tools = {
            "get_anomaly_score": tools.get_anomaly_score,
            "get_user_history": tools.get_user_history,
            "analyze_contextual_patterns": tools.analyze_contextual_patterns,
        }
        self.decision_log = []

    def _initialize_hf_client(self, model, token):
        """Initializes and returns the Hugging Face Inference Client."""
        try:
            client = InferenceClient(model=model, token=token)
            logger.info(f"Initialized Inference API client for {model}.")
            return client
        except Exception as e:
            logger.error(f"Failed to initialize Inference API client: {e}")
            return None

    def _build_prompt(self, transaction, history):
        """Builds the system prompt for the LLM, defining its role, tools, and the task."""
        # Basic transaction details
        transaction_details = (
            f"Time: {transaction['Time'].iloc[0]:.2f}, "
            f"Amount: {transaction['Amount'].iloc[0]:.2f}, "
            f"UserID: {transaction['user_id'].iloc[0]}"
        )

        # Tool definitions
        tool_definitions = """
        Available Tools:
        - `get_anomaly_score(transaction)`: Returns 'High Anomaly Score' or 'Normal Score' from a machine learning model.
        - `get_user_history(user_id, transaction_amount)`: Returns a summary of the user's past transaction behavior.
        - `analyze_contextual_patterns(transaction_time, transaction_amount)`: Checks for suspicious patterns like unusual times or round numbers.
        """

        # The core instruction prompt
        prompt = f"""
        You are a fraud detection expert. Your goal is to determine if a credit card transaction is fraudulent.
        You must use the available tools to gather evidence and build a case.
        Do not make a final decision until you have gathered sufficient evidence.

        **Transaction Details:**
        {transaction_details}

        **Instructions:**
        1.  **Think**: Analyze the situation and decide which tool will provide the most useful information.
        2.  **Act**: Choose ONE tool and its parameters.
        3.  **Observe**: You will receive the output from the tool.
        4.  Repeat this process for a few steps, accumulating evidence.
        5.  When you have enough evidence, provide a `Final Answer` which is either "Flag" or "Approve", followed by a concise summary of your reasoning.

        **Example Interaction:**

        **Thought**: I should first check the baseline anomaly score to see if it's a clear case of fraud.
        **Action**: `get_anomaly_score(transaction)`

        **Observation**: High Anomaly Score (Potential Fraud)

        **Thought**: The score is high. I need to see if this transaction is unusual for this specific user.
        **Action**: `get_user_history(user_id=101, transaction_amount=500.0)`

        **Observation**: Transaction amount is significantly higher than user's average of $50.00.

        **Thought**: High anomaly score and unusual amount for the user. This is strong evidence. I will make a final decision.
        **Final Answer**: Flag. The transaction has a high anomaly score from the model and the amount is far greater than the user's typical spending pattern.

        Begin!

        **History of this interaction:**
        {history}
        """
        return prompt.strip()

    def process_transaction(self, transaction: pd.Series, user_id: str):
        """
        Processes a single transaction using the ReAct framework.
        The LLM orchestrates calls to various tools to gather evidence and make a final decision.
        """
        if self.client is None:
            return "Error: Hugging Face client not initialized.", []

        # Add the selected user_id to the transaction data
        transaction["user_id"] = user_id
        
        # Convert Series to DataFrame for tool compatibility
        transaction_df = transaction.to_frame().T

        history = ""
        full_reasoning_chain = []

        for _ in range(3):  # Limit to 3 steps to prevent infinite loops
            prompt = self._build_prompt(transaction_df, history)

            try:
                response_text = (
                    self.client.chat_completion(
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=200,
                        temperature=0.5,
                    )
                    .choices[0]
                    .message.content
                )
                if not response_text:
                    raise ValueError("API returned an empty response.")
            except Exception as e:
                logger.error(f"Error calling HF API: {e}")
                return f"Error: API call failed. Details: {e}", full_reasoning_chain

            full_reasoning_chain.append(response_text)
            history += f"\n{response_text}"

            if "Final Answer:" in response_text:
                final_answer = response_text.split("Final Answer:")[-1].strip()
                action = "Flag" if "Flag" in final_answer else "Approve"
                self._log_decision(transaction_df, action, full_reasoning_chain)
                return action, full_reasoning_chain

            # Parse the action and execute the tool
            try:
                action_str = response_text.split("Action:")[-1].strip()
                tool_name = action_str.split("(")[0]

                # This is a simplified parser. A real implementation would be more robust.
                if tool_name in self.tools:
                    # Simplified argument extraction
                    if tool_name == "get_anomaly_score":
                        observation = self.tools[tool_name](transaction_df)
                    elif tool_name == "get_user_history":
                        observation = self.tools[tool_name](
                            user_id=transaction["user_id"], transaction_amount=transaction["Amount"]
                        )
                    elif tool_name == "analyze_contextual_patterns":
                        observation = self.tools[tool_name](
                            transaction_time=transaction["Time"], transaction_amount=transaction["Amount"]
                        )
                    else:
                        observation = "Unknown tool."
                else:
                    observation = f"Error: Tool '{tool_name}' not found."
            except Exception as e:
                observation = f"Error parsing or executing action: {e}"

            history += f"\n**Observation**: {observation}"
            full_reasoning_chain.append(f"Observation: {observation}")

        self._log_decision(transaction_df, "Inconclusive", full_reasoning_chain)
        return "Inconclusive: Max steps reached.", full_reasoning_chain

    def _log_decision(self, features, action, reasoning_chain):
        """Logs the final decision and the full reasoning chain."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "transaction": features.to_dict("records")[0],
            "decision": action,
            "reasoning": reasoning_chain,
            "threshold": "N/A",  # Threshold is no longer the deciding factor
        }
        self.decision_log.append(log_entry)
        decision_logger.info(f"Action: {action}, Reasoning: {' -> '.join(reasoning_chain)}")


def main():
    """Main function to test the new ReAct Agent."""
    hf_token = os.getenv("HF_TOKEN")
    hf_model = os.getenv("HF_INFERENCE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

    agent = ReActAgent(hf_token=hf_token, hf_model=hf_model)

    # Load test data
    try:
        test_df = pd.read_csv("data/processed/test.csv")
    except FileNotFoundError:
        logger.error("Error: 'test.csv' not found. Run preprocess.py first.")
        exit(1)

    # Process a single fraudulent transaction for a clear test case
    fraudulent_transaction = test_df[test_df["Class"] == 1].iloc[0]
    
    logger.info("Processing a known fraudulent transaction for user 102...")
    action, reasoning = agent.process_transaction(fraudulent_transaction, user_id="102")
    logger.info(f"Final Action: {action}")
    logger.info("Full Reasoning Chain:")
    for step in reasoning:
        logger.info(step)

    # Save the decision log
    try:
        pd.DataFrame(agent.decision_log).to_csv(os.path.join(LOG_DIR, "decision_log.csv"), index=False)
        logger.info("Decision log saved to logs/decision_log.csv.")
    except Exception as e:
        logger.error(f"Error saving decision log: {e}")


if __name__ == "__main__":
    main()
