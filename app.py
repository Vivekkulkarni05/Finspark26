import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
from streamlit_agraph import agraph, Node, Edge, Config

import model_trainer
import explainability
import quantum_scorer
import os

st.set_page_config(page_title="QuantumGuard", layout="wide")

st.title("QuantumGuard Prototype")
st.markdown("""
**Data Source:** Synthetic dataset based on PaySim and CICIDS2017 schemas, incorporating fabricated shared identifiers for demonstration.
""")

@st.cache_data
def load_data():
    if not os.path.exists("joined_data.csv"):
        return None
    return pd.read_csv("joined_data.csv")

@st.cache_resource
def get_ml_components(df):
    X, y = model_trainer.prepare_data(df)
    model, metrics, X_train, X_test, y_test = model_trainer.train_model(X, y)
    explainer = explainability.get_explainer(model, X_train)
    return model, metrics, explainer, X.columns.tolist(), X

df = load_data()
if df is None:
    st.error("Data not found. Please run data_generator.py first.")
    st.stop()

model, metrics, explainer, feature_names, X = get_ml_components(df)

# Prepare alerts
if 'predictions' not in st.session_state:
    st.session_state.predictions = pd.DataFrame()

# We score the entire dataframe for visualization
probs = model.predict_proba(X)[:, 1]
df['fraud_score'] = probs
df['xgboost_flag'] = df['fraud_score'] > 0.5

# Calculate Quantum Score & Hub Nodes
if 'processed_alerts' not in st.session_state:
    quantum_scores = []
    quantum_breakdowns = []
    for idx, row in df.iterrows():
        s, b = quantum_scorer.score_quantum_exposure(row)
        quantum_scores.append(s)
        quantum_breakdowns.append(b)
    
    df['quantum_score'] = quantum_scores
    df['quantum_breakdown'] = quantum_breakdowns
    
    # Graph Centrality (simplified)
    G = nx.Graph()
    for _, row in df.iterrows():
        user = str(row['user_id'])
        dev = str(row['device_id'])
        acc = str(row['account_id'])
        ip = str(row['ip'])
        G.add_edges_from([(user, dev), (user, acc), (dev, ip)])
    
    degree_dict = dict(G.degree())
    df['hub_node_flag'] = df['device_id'].map(lambda x: degree_dict.get(str(x), 0) > 5)
    
    # Ensemble logic
    df['high_confidence_alert'] = df['xgboost_flag'] & (df['hub_node_flag'] | (df['quantum_score'] > 30))
    
    st.session_state.processed_alerts = df
else:
    df = st.session_state.processed_alerts

# Feedback loop tracking
if 'false_positives' not in st.session_state:
    if os.path.exists('fp_feedback.csv'):
        st.session_state.false_positives = pd.read_csv('fp_feedback.csv')['event_id_x'].tolist()
    else:
        st.session_state.false_positives = []

def mark_fp(event_id):
    if event_id not in st.session_state.false_positives:
        st.session_state.false_positives.append(event_id)
        pd.DataFrame({'event_id_x': st.session_state.false_positives}).to_csv('fp_feedback.csv', index=False)
        st.toast(f"Marked {event_id} as False Positive")

tab1, tab2, tab3, tab4 = st.tabs(["Alert Feed", "Correlation Graph", "Model Metrics", "Quantum Exposure Assessment"])

with tab1:
    st.subheader("High Confidence Alerts")
    
    alerts = df[df['high_confidence_alert'] == True].copy()
    alerts = alerts[~alerts['event_id_x'].isin(st.session_state.false_positives)]
    alerts = alerts.sort_values(by=['fraud_score', 'quantum_score'], ascending=[False, False])
    
    for idx, row in alerts.head(20).iterrows():
        with st.expander(f"Alert: {row['event_id_x']} | Fraud Score: {row['fraud_score']:.2f} | Quantum Score: {row['quantum_score']}"):
            cols = st.columns(3)
            with cols[0]:
                st.write("**Entities Involved:**")
                st.write(f"- User: {row['user_id']}")
                st.write(f"- Device: {row['device_id']}")
                st.write(f"- Account: {row['account_id']}")
            with cols[1]:
                st.write("**Event Details:**")
                st.write(f"- Amount: ${row['amount']}")
                st.write(f"- Type: {row['type']}")
                st.write(f"- TLS: {row['tls_version']}")
            with cols[2]:
                st.button("Mark as False Positive", key=f"fp_{row['event_id_x']}", on_click=mark_fp, args=(row['event_id_x'],))
            
            # Explainability
            st.write("---")
            st.write("**AI Explanation:**")
            row_features = X.loc[[idx]]
            explanation = explainability.explain_alert(explainer, row_features, feature_names)
            st.info(explanation)
            
            if row['hub_node_flag']:
                st.warning("Flagged: Node exhibits hub-like behavior in correlation graph (Potential coordinated activity detected).")
            if row['quantum_score'] > 0:
                st.warning(f"Quantum Exposure Factors: {', '.join(row['quantum_breakdown'])}")

with tab2:
    st.subheader("Entity Correlation Graph")
    st.write("Visualizing a subset of the graph to highlight potential coordinated activity and shared identifiers.")
    
    # Filter for interesting nodes to keep graph manageable
    interesting_alerts = df[(df['hub_node_flag'] == True) | (df['event_id_x'].str.startswith('TX-RING'))]
    if len(interesting_alerts) == 0:
        interesting_alerts = df.head(50)
        
    G_viz = nx.Graph()
    for _, row in interesting_alerts.iterrows():
        user = str(row['user_id'])
        dev = str(row['device_id'])
        acc = str(row['account_id'])
        ip = str(row['ip'])
        
        # Color code nodes based on type
        G_viz.add_node(user, group="user", title=f"User: {user}")
        G_viz.add_node(dev, group="device", title=f"Device: {dev}")
        G_viz.add_node(acc, group="account", title=f"Account: {acc}")
        G_viz.add_node(ip, group="ip", title=f"IP: {ip}")
        
        G_viz.add_edges_from([(user, dev), (user, acc), (dev, ip)])

    nodes = []
    edges = []
    
    # Simple color mapping
    color_map = {"user": "#3498db", "device": "#e74c3c", "account": "#2ecc71", "ip": "#f1c40f"}
    
    for node, attrs in G_viz.nodes(data=True):
        group = attrs.get('group', 'user')
        nodes.append(Node(id=node, label=node, size=25, color=color_map.get(group)))
        
    for source, target in G_viz.edges():
        edges.append(Edge(source=source, target=target))
        
    config = Config(width=800, height=600, directed=False, physics=True, hierarchical=False)
    
    agraph(nodes=nodes, edges=edges, config=config)

with tab3:
    st.subheader("Model Performance (XGBoost)")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AUC", f"{metrics['auc']:.4f}")
    col2.metric("Precision", f"{metrics['precision']:.4f}")
    col3.metric("Recall", f"{metrics['recall']:.4f}")
    col4.metric("F1 Score", f"{metrics['f1']:.4f}")
    
    st.write("---")
    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(metrics['cm'], index=['Actual Neg', 'Actual Pos'], columns=['Pred Neg', 'Pred Pos'])
    st.dataframe(cm_df)

with tab4:
    st.subheader("Future Quantum Exposure Assessment")
    st.write("Rule-based heuristic evaluation utilizing cryptographic patterns and anomalous data transfer metrics.")
    
    q_alerts = df[df['quantum_score'] > 0].sort_values(by='quantum_score', ascending=False)
    
    st.dataframe(
        q_alerts[['event_id_x', 'timestamp', 'user_id', 'quantum_score', 'quantum_breakdown', 'tls_version', 'cipher_suite', 'bytes_sent']]
        .head(20),
        use_container_width=True
    )
