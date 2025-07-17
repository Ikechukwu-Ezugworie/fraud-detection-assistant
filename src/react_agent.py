import logging
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.tools import all_tools

load_dotenv()

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

gemini_api_key = os.getenv("GEMINI_API_KEY")
gemini_model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")


class ReActAgent:
    def __init__(self):
        """Initializes the ReAct agent using LangChain and a Hugging Face model."""
        self.tools = all_tools
        self.tool_map = {tool.name: tool for tool in all_tools}
        self.llm = self._initialize_llm()
        self.decision_log = []

    def _initialize_llm(self):
        """Initializes and returns the  LLM."""
        try:
            llm = ChatGoogleGenerativeAI(
                model="models/gemini-1.5-flash-latest",
                google_api_key=gemini_api_key,
                temperature=0,
            )

            llm = llm.bind_tools(self.tools)

            logger.info(f"Initialized  LLM for {gemini_model}.")
            return llm
        except Exception as e:
            logger.error(f"Failed to initialize  LLM: {e}")
            return None

    def process_transaction(self, transaction: pd.Series):
        history: list[BaseMessage] = []
        max_iterations = 3
        system_message = (
            "You are a helpful assistant that analyzes financial transactions. Make use of the tools provided."
            "To analyze a transaction, you need to know the anomaly score"
            "Using the anomaly score, you can get decision with reasoning"
        )
        history.append(SystemMessage(system_message))
        history.append(HumanMessage(f"Analyze the following transaction: {transaction.to_dict()}"))

        # if self.llm:
        #     for i in range(max_iterations):

        #         response = self.llm.invoke(history)

        #         if not response.tool_calls:  # type: ignore
        #             history.append(SystemMessage("No further action needed."))
        #             break

        #         for tool_call in response.tool_calls:  # type: ignore
        #             tool = self.tool_map.get(tool_call["name"])

        #             if tool:
        #                 tool_output = tool.invoke(tool_call["args"])

        #                 history.append(ToolMessage(content=tool_output, tool_call_id=tool_call["id"]))
        #         time.sleep(3)
        # return history
        anomaly_tool = self.tool_map.get("get_anomaly_score")
        anomaly_score = anomaly_tool.invoke(input={"features": transaction.to_dict()}) if anomaly_tool else 0
        history.append(ToolMessage(content=f"Anomaly score: {anomaly_score}", tool_call_id="anomaly_score"))

        decision_tool = self.tool_map.get("get_decision_with_reasoning")
        decision_inputs = {
            "amount": transaction["Amount"],
            "time": transaction["Time"],
            "anomaly_score": anomaly_score,
        }
        reasoning = decision_tool.invoke(decision_inputs) if decision_tool else ""
        history.append(ToolMessage(content=f"Reasoning: {reasoning}", tool_call_id="reasoning"))

        return history

    def _log_decision(self, features, action, reasoning_chain):
        """Logs the final decision and reasoning."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "transaction": features.to_dict("records")[0],
            "decision": action,
            "reasoning": reasoning_chain,
        }
        self.decision_log.append(log_entry)
        decision_logger.info(f"Action: {action}, Reasoning: {reasoning_chain}")


def main():
    """Main function to test the new LangChain-based ReAct Agent."""
    agent = ReActAgent()

    # Load test data
    try:
        test_df = pd.read_csv("data/processed/test.csv")
    except FileNotFoundError:
        logger.error("Error: 'test.csv' not found. Run preprocess.py first.")
        exit(1)

    # Process a known fraudulent transaction
    fraudulent_transaction = test_df[test_df["Class"] == 1].iloc[0].drop("Class")

    logger.info("Processing a known fraudulent transaction...")
    history = agent.process_transaction(fraudulent_transaction)
    logger.info(f"History: \n\n{history}")

    # Save the decision log
    try:
        pd.DataFrame(agent.decision_log).to_csv(os.path.join(LOG_DIR, "decision_log.csv"), index=False)
        logger.info("Decision log saved to logs/decision_log.csv.")
    except Exception as e:
        logger.error(f"Error saving decision log: {e}")


if __name__ == "__main__":
    main()
