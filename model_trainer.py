import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import xgboost as xgb

def prepare_data(df, false_positives=None):
    """
    Prepares the joined dataframe for modeling by selecting features,
    handling categoricals, and separating the target.
    """
    if false_positives:
        df.loc[df['event_id_x'].isin(false_positives), 'isFraud'] = 0

    # Select features
    numeric_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'flow_duration', 'bytes_sent', 'bytes_received', 'user_tx_velocity']
    categorical_features = ['type', 'protocol', 'tls_version', 'cipher_suite']
    
    X = df[numeric_features + categorical_features].copy()
    y = df['isFraud'].copy()
    
    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=categorical_features, dummy_na=False)
    
    # Fill any remaining NaNs and ensure everything is float for SHAP compatibility
    X = X.fillna(0).astype(float)
    
    return X, y

def train_model(X, y):
    """
    Trains an XGBoost classifier on the data and returns the model,
    test features (for SHAP), and performance metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    metrics = {
        'auc': roc_auc_score(y_test, y_prob),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'cm': confusion_matrix(y_test, y_pred)
    }
    
    return model, metrics, X_train, X_test, y_test

if __name__ == "__main__":
    df = pd.read_csv("joined_data.csv")
    X, y = prepare_data(df)
    model, metrics, X_train, X_test, y_test = train_model(X, y)
    print("Metrics:", metrics)