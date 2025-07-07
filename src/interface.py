import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import logging


# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
from src.react_agent import ReActAgent

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check Streamlit version
if st.__version__ < "1.38.0":
    st.warning("Streamlit version <1.38.0 detected. Please upgrade to 1.38.0 or higher for optimal performance.",
               icon="⚠️")
    logger.warning(f"Streamlit version {st.__version__} detected, expected >=1.38.0")

# Page configuration (no sidebar)
st.set_page_config(page_title="Fraud Detection Assistant Agent", layout="wide", initial_sidebar_state="collapsed")

# Add left and right margins
st.markdown(
    """
    <style>
    .stApp {
        margin-left: 5vw;
        margin-right: 5vw;
    }
    @media (max-width: 600px) {
        .stApp {
            margin-left: 2vw;
            margin-right: 2vw;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Initialize ReAct agent
@st.cache_resource
def load_agent():
    try:
        agent = ReActAgent(hf_token=None)
        logger.info("ReAct agent loaded successfully.")
        return agent
    except Exception as e:
        logger.error(f"Error loading ReAct agent: {str(e)}")
        st.error("Failed to load ReAct agent. Check logs for details.")
        return None


# Load decision log with pagination support
@st.cache_data
def load_decision_log(page=1, page_size=50, filter_flagged=False):
    log_path = 'logs/decision_log.csv'
    try:
        if os.path.exists(log_path):
            df = pd.read_csv(log_path)
            if filter_flagged:
                df = df[df['decision'] == 'Flag']
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            logger.info(f"Loaded decision log from {log_path}, page {page}, filter_flagged={filter_flagged}")
            return df.iloc[start_idx:end_idx], len(df)
        else:
            logger.warning(f"Decision log {log_path} not found.")
            return pd.DataFrame(), 0
    except Exception as e:
        logger.error(f"Error loading decision log: {str(e)}")
        return pd.DataFrame(), 0


# Parse transaction features
def parse_transaction_features(transaction_str):
    try:
        transaction = eval(transaction_str) if isinstance(transaction_str, str) else transaction_str
        return {k: f"{v:.2f}" for k, v in transaction.items()}
    except:
        return {}


# Convert threshold to confidence score
def threshold_to_confidence(threshold):
    max_threshold = 0.0017
    min_threshold = 0.0005
    return 1 - (threshold - min_threshold) / (max_threshold - min_threshold)


# Main app
def main():
    st.title("🛡️ Fraud Detection Assistant Agent")
    st.markdown("Detect credit card fraud with AI-powered reasoning and interactive visualizations.")

    # Initialize agent
    agent = load_agent()
    if agent is None:
        return

    # Guided tour with responsive animation
    if 'tour_seen' not in st.session_state:
        st.session_state['tour_seen'] = False
    if not st.session_state['tour_seen']:
        with st.container():
            st.markdown(
                """
                <style>
                @keyframes fadeIn {
                    0% { opacity: 0; }
                    100% { opacity: 1; }
                }
                .tour-modal {
                    animation: fadeIn 1s;
                    border: 1px solid #ddd;
                    padding: 2vw;
                    border-radius: 10px;
                    max-width: 90vw;
                    margin: 0 auto;
                    font-size: calc(0.8rem + 0.5vw);
                }
                @media (max-width: 600px) {
                    .tour-modal {
                        padding: 4vw;
                        font-size: calc(0.7rem + 1vw);
                    }
                }
                </style>
                <div class="tour-modal">
                    <h3>Welcome to the Fraud Detection Assistant Agent!</h3>
                    <p>This dashboard helps you detect credit card fraud with AI. Here's a quick tour:</p>
                    <ul>
                        <li><b>Overview</b>: See fraud rates and recent decisions.</li>
                        <li><b>Historical Analysis</b>: Explore past transactions with charts.</li>
                        <li><b>Test Transactions</b>: Analyze new transactions in real-time.</li>
                        <li><b>FAQ</b>: Learn about the system’s components.</li>
                    </ul>
                    <p>Click 'Got it!' to start.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

            if st.button("Got it!", use_container_width=False):
                st.session_state['tour_seen'] = True
                try:
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error during rerun: {str(e)}")
                    st.error("Failed to refresh the page. Please refresh manually.")

    # Tabs for navigation
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Historical Analysis", "Test Transactions", "FAQ"])

    # Tab 1: Overview
    with tab1:
        st.header("Dashboard Overview", anchor="overview")
        st.markdown("""
        This lightweight fraud detection assistant uses the ReAct framework to identify suspicious credit card transactions from the [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 
        It employs logic-based reasoning to flag anomalies without manual reconfiguration, combining an Isolation Forest model with Gemma 2 9B for AI-driven insights. 
        Use the tabs to explore historical decisions, test new transactions, or learn more about the system.
        """)

        decision_log, total_rows = load_decision_log()
        if not decision_log.empty:
            fraud_ratio = (decision_log['decision'] == 'Flag').mean() * 100
            col1, col2, col3 = st.columns([1, 1, 1], gap="small")
            with col1:
                st.metric("Total Transactions", total_rows)
            with col2:
                st.metric("Fraud Rate", f"{fraud_ratio:.2f}%", f"{(decision_log['decision'] == 'Flag').sum()} flagged")
            with col3:
                avg_threshold = decision_log['threshold'].mean()
                st.metric("Average Threshold", f"{avg_threshold:.6f}")

            st.subheader("Latest Decision")
            latest_decision = decision_log.tail(1)
            if not latest_decision.empty:
                action = latest_decision['decision'].iloc[0]
                reasoning = latest_decision['reasoning'].iloc[0].strip('[]')
                threshold = latest_decision['threshold'].iloc[0]
                confidence = threshold_to_confidence(threshold)
                if action == "Flag":
                    st.error(f"**Decision**: {action}", icon="🚨")
                else:
                    st.success(f"**Decision**: {action}", icon="✅")
                st.markdown(f"**Reasoning**: {reasoning}")
                st.markdown(f"**Confidence**: {confidence:.2%}")
                st.markdown(f"**Timestamp**: {latest_decision['timestamp'].iloc[0]}")

    # Tab 2: Historical Analysis
    with tab2:
        st.header("Historical Transaction Analysis", anchor="historical-analysis")
        st.markdown("Explore past decisions with filterable tables and visualizations.")

        # Pagination and filter controls
        col1, col2 = st.columns([3, 1], gap="small")
        with col1:
            filter_flagged = st.checkbox("Show only flagged transactions", value=False)
        with col2:
            page_size = 50  # Fixed for simplicity
            total_rows = load_decision_log(filter_flagged=filter_flagged)[1]
            page = st.selectbox("Page", options=list(range(1, int(np.ceil(total_rows / page_size)) + 1)), index=0)

        decision_log, total_rows = load_decision_log(page, page_size, filter_flagged)
        if not decision_log.empty:
            # Decision log table
            st.subheader("Decision Log")
            decision_log['features'] = decision_log['transaction'].apply(parse_transaction_features)
            st.dataframe(
                decision_log[['timestamp', 'decision', 'reasoning', 'threshold', 'features']],
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Timestamp"),
                    "decision": st.column_config.TextColumn("Decision",
                                                            help="Flag: Potential fraud; Approve: Normal transaction."),
                    "reasoning": st.column_config.ListColumn("Reasoning",
                                                             help="AI-generated or rule-based explanation."),
                    "threshold": st.column_config.NumberColumn("Threshold", format="%.6f",
                                                               help="Anomaly detection threshold."),
                    "features": st.column_config.TextColumn("Features",
                                                            help="Normalized transaction features (Time, Amount, V1-V28).")
                },
                use_container_width=True
            )
            st.markdown(f"Showing page {page} of {int(np.ceil(total_rows / page_size))}")

            # Download button
            csv = decision_log.to_csv(index=False)
            st.download_button(
                label="Download Current Page of CSV",
                data=csv,
                file_name=f"decision_log_page_{page}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=False
            )
            st.markdown("---")

            # Create 3 columns: chart | separator | chart
            col1, col_sep, col2 = st.columns([5, 0.2, 5])  # Separator takes minimal space

            # Column 1: Pie Chart
            with col1:
                st.markdown("### Decision Distribution")
                decision_counts = decision_log['decision'].value_counts(normalize=True) * 100
                df = pd.DataFrame({
                    "Decision": decision_counts.index,
                    "Count": decision_counts.values
                })

                fig_pie = px.pie(
                    data_frame=df,
                    names="Decision",
                    values="Count",
                    color="Decision",
                    hover_data=["Count"],
                    title="Flagged vs. Approved Transactions",
                    color_discrete_map={"Flag": "red", "Approve": "green"}
                )

                fig_pie.update_traces(
                    textinfo="percent+label",
                    hovertemplate="%{label}: %{value:.2f}%",
                    pull=[0.1 if x == "Flag" else 0 for x in decision_counts.index]
                )
                fig_pie.update_layout(height=400, margin=dict(t=50, b=50, l=20, r=20))
                st.plotly_chart(fig_pie, use_container_width=True)

            # Column separator: vertical line using HTML
            with col_sep:
                st.markdown(
                    """
                    <div style="
                        width: 1px;
                        height: 400px;
                        background-color: #BBB;
                        margin-left: auto;
                        margin-right: auto;
                    "></div>
                    """,
                    unsafe_allow_html=True
                )

            # Column 2: Scatter Plot
            with col2:
                st.markdown("### Transaction Patterns")
                fig_scatter = px.scatter(
                    decision_log,
                    x="timestamp",
                    y=decision_log['transaction'].apply(
                        lambda x: eval(x)['Amount'] if isinstance(x, str) else x['Amount']),
                    color="decision",
                    color_discrete_map={"Flag": "red", "Approve": "green"},
                    title="Transaction Amount Over Time",
                    labels={"y": "Amount", "x": "Timestamp"},
                    hover_data=["reasoning", "threshold"]
                )
                fig_scatter.update_traces(marker=dict(size=10), selector=dict(mode='markers'))
                fig_scatter.update_layout(height=400, margin=dict(t=50, b=50, l=20, r=20))
                st.plotly_chart(fig_scatter, use_container_width=True)

            st.markdown("---")

            # Optimized heatmap
            st.subheader("Feature Correlations in Flagged Transactions")
            flagged = decision_log[decision_log['decision'] == 'Flag']
            if not flagged.empty:
                features = pd.DataFrame([eval(x) if isinstance(x, str) else x for x in flagged['transaction']])
                selected_cols = ['Time', 'Amount', 'V1', 'V10', 'V14']
                corr = features[selected_cols].corr().round(2)

                annotations = []
                for i in range(len(corr)):
                    for j in range(len(corr.columns)):
                        annotations.append(
                            dict(
                                x=corr.columns[j],
                                y=corr.index[i],
                                text=str(corr.iloc[i, j]),
                                showarrow=False,
                                font=dict(color="white" if abs(corr.iloc[i, j]) > 0.5 else "black")
                            )
                        )

                fig_heatmap = go.Figure(data=go.Heatmap(
                    z=corr.values,
                    x=corr.columns,
                    y=corr.index,
                    colorscale='Viridis',  # Plotly default
                    colorbar=dict(title="Correlation"),
                    hovertemplate="%{x} vs %{y}: %{z:.2f}"
                ))

                fig_heatmap.update_layout(
                    title="Correlation Heatmap for Flagged Transactions",
                    height=500,
                    annotations=annotations,
                    xaxis=dict(tickangle=-45),
                    margin=dict(t=60, b=60, l=40, r=40),
                    font=dict(size=12)
                )

                st.plotly_chart(fig_heatmap, use_container_width=True)
            else:
                st.info("No flagged transactions available for case study.")

    # Tab 3: Test Transactions
    with tab3:
        st.header("Test a New Transaction", anchor="test-transactions")
        st.markdown("Analyze a transaction for fraud in real-time or upload a CSV for bulk testing.")

        # API rate limit warning
        try:
            response = agent.test_hf_api()
            if "rate limit" in str(response).lower():
                st.warning(
                    "Hugging Face API rate limit may be reached. Consider reducing transaction frequency or checking API status.",
                    icon="⚠️")
        except:
            st.warning("Unable to connect to Hugging Face API. Results may use rule-based reasoning.", icon="⚠️")

        # File upload
        uploaded_file = st.file_uploader("Upload Transaction CSV", type="csv",
                                         help="CSV with columns: Time, Amount, V1-V28")
        if uploaded_file:
            try:
                transactions = pd.read_csv(uploaded_file)
                expected_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                if not all(col in transactions.columns for col in expected_columns):
                    st.error("CSV must contain columns: Time, Amount, V1-V28")
                else:
                    with st.spinner("Processing transactions..."):
                        results = []
                        for _, row in transactions.iterrows():
                            action = agent.process_transaction(row)
                            decision_log = load_decision_log()[0]
                            latest = decision_log.tail(1)
                            results.append({
                                "Decision": action,
                                "Reasoning": latest['reasoning'].iloc[0].strip('[]') if not latest.empty else "N/A",
                                "Confidence": threshold_to_confidence(
                                    latest['threshold'].iloc[0]) if not latest.empty else 0
                            })
                        st.dataframe(pd.DataFrame(results), use_container_width=True)
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")

        # Manual input form
        with st.form("transaction_form"):
            col1, col2 = st.columns([1, 1], gap="small")
            with col1:
                time_input = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0,
                                             step=0.1, help="Normalized time since the first transaction.")
                amount_input = st.number_input("Amount (normalized)", min_value=0.0, value=0.0, step=0.01,
                                               help="Normalized transaction amount.")

            with col2:
                st.write("Key Features (PCA-transformed)")
                with st.expander("Enter V1-V28 Features"):
                    v_inputs = {f"V{i}": st.number_input(f"V{i}", value=0.0, step=0.01, key=f"v{i}") for i in
                                range(1, 29)}

            col_btn1, col_btn2 = st.columns([1, 1], gap="small")
            with col_btn1:
                if st.form_submit_button("Randomize Features", use_container_width=True):
                    time_input = np.random.uniform(0, 1)
                    amount_input = np.random.uniform(-0.5, 0.5)
                    v_inputs = {f"V{i}": np.random.uniform(-5, 5) for i in range(1, 29)}
                    st.session_state['time_input'] = time_input
                    st.session_state['amount_input'] = amount_input
                    st.session_state['v_inputs'] = v_inputs
                    try:
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Error during rerun: {str(e)}")
                        st.error("Failed to refresh the page. Please refresh manually.")

            with col_btn2:
                submitted = st.form_submit_button("Analyze Transaction", type="primary", use_container_width=True)
                if submitted:
                    if agent:
                        try:
                            transaction = pd.DataFrame([{
                                "Time": time_input,
                                **{f"V{i}": v_inputs[f"V{i}"] for i in range(1, 29)},
                                "Amount": amount_input
                            }])
                            with st.spinner("Analyzing transaction..."):
                                action = agent.process_transaction(transaction.iloc[0])
                                decision_log = load_decision_log()[0]
                                latest_decision = decision_log.tail(1)

                            if not latest_decision.empty:
                                reasoning = latest_decision['reasoning'].iloc[0].strip('[]')
                                threshold = latest_decision['threshold'].iloc[0]
                                confidence = threshold_to_confidence(threshold)
                                st.subheader("Result")
                                if action == "Flag":
                                    st.error(f"**Decision**: {action}", icon="🚨")
                                else:
                                    st.success(f"**Decision**: {action}", icon="✅")
                                st.markdown(f"**Reasoning**: {reasoning}")
                                st.markdown(f"**Confidence**: {confidence:.2%}")
                                st.markdown(f"**Timestamp**: {latest_decision['timestamp'].iloc[0]}")
                                st.markdown("**Key Features**")
                                st.write(f"- Amount: {transaction['Amount'].iloc[0]:.2f}")
                                st.write(f"- V10: {transaction['V10'].iloc[0]:.2f}")
                                st.write(f"- V14: {transaction['V14'].iloc[0]:.2f}")
                        except Exception as e:
                            st.error(f"Error processing transaction: {str(e)}")
                    else:
                        st.error("ReAct agent not initialized.")

    # Tab 4: FAQ
    with tab4:
        st.header("Frequently Asked Questions", anchor="faq")
        st.markdown("Learn more about the Fraud Detection Assistant.")
        with st.expander("What is an Isolation Forest?"):
            st.markdown(
                "An Isolation Forest is a machine learning algorithm that identifies anomalies by isolating data points in a dataset. It works by randomly splitting features, making it efficient for detecting outliers like fraudulent transactions.")
        with st.expander("What are V1-V28 features?"):
            st.markdown(
                "V1-V28 are anonymized features derived from Principal Component Analysis (PCA) applied to the original transaction data. They represent patterns in the data while protecting sensitive information.")
        with st.expander("How does the ReAct agent work?"):
            st.markdown(
                "The ReAct agent combines the Isolation Forest model with Gemma 2 9B, an AI model that generates natural language explanations. It observes transaction features, reasons about anomalies, decides to flag or approve, and logs the results.")
        with st.expander("What is the confidence score?"):
            st.markdown(
                "The confidence score reflects the reliability of the decision, derived from the adaptive threshold used by the Isolation Forest. Lower thresholds indicate higher confidence in flagging anomalies.")


if __name__ == "__main__":
    main()