import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
from io import StringIO
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb

import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import scipy.sparse


# ----------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------

app = FastAPI(title="Full Churn Prediction Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend from any port (3000, 3889, etc.)
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


# ----------------------------------------------------
# GLOBAL STORAGE FOR VISUALIZATION DATA
# ----------------------------------------------------

viz_cache = {
    "model": None,
    "X_test": None,
    "y_test": None,
    "y_pred": None,
    "y_proba": None,
    "df_clean": None,
}


# ----------------------------------------------------
# RESPONSE MODEL
# ----------------------------------------------------

class PredictionResponse(BaseModel):
    model_used: str
    training_accuracy: float
    testing_accuracy: float
    roc_auc: float
    mean_prediction_probability: float
    confusion_matrix: list
    classification_report: str
    business_summary: str


# ----------------------------------------------------
# HELPER: MATPLOTLIB FIGURE → BASE64 STRING
# ----------------------------------------------------

def fig_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    return encoded


# ----------------------------------------------------
# MODEL SELECTOR
# ----------------------------------------------------

def get_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=3000)

    if model_name == "naive_bayes":
        return GaussianNB()

    if model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5)

    if model_name == "svm":
        return SVC(probability=True)

    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=42)

    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=300, random_state=42)

    if model_name == "xgboost":
        return xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )

    if model_name == "lightgbm":
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=40,
            random_state=42,
        )

    raise ValueError(f"Unknown model: {model_name}")


# ----------------------------------------------------
# DATA CLEANING FOR TELCO-STYLE DATASET
# ----------------------------------------------------

def clean_telco(df: pd.DataFrame, target_col: str):

    # Strip spaces for all string columns
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()


    # Fix TotalCharges if present
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Normalize "No internet service"/"No phone service"
    df.replace(
        {"No internet service": "No", "No phone service": "No"},
        inplace=True,
    )

    # SeniorCitizen → Yes/No
    if "SeniorCitizen" in df.columns and df["SeniorCitizen"].dtype != "O":
        df["SeniorCitizen"] = df["SeniorCitizen"].replace({0: "No", 1: "Yes"})

    # Target column processing (e.g., Churn)
    if target_col in df.columns:
        # Treat "Yes"/"No" as 1/0 if present
        if df[target_col].dtype == "O":
            df[target_col] = df[target_col].replace({"Yes": 1, "No": 0})

    # Drop ID-like columns that are not useful for prediction
    id_columns = ["customerID", "CustomerID", "ID", "id", "accountID"]
    drop_ids = [c for c in id_columns if c in df.columns]
    if drop_ids:
        df = df.drop(columns=drop_ids)

    return df


# ----------------------------------------------------
# TRAIN + EVALUATE FUNCTION
# ----------------------------------------------------

def train_and_evaluate(df: pd.DataFrame, target_col: str, model_name: str):

    df = clean_telco(df, target_col)

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[("scale", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("encode", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    model = get_model(model_name)

    # Models that need dense input
    needs_dense = model_name.lower() in ["naive_bayes", "knn", "svm"]

    if needs_dense:
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "to_dense",
                    FunctionTransformer(
                        lambda x: x.toarray() if scipy.sparse.issparse(x) else x
                    ),
                ),
                ("model", model),
            ]
        )
    else:
        clf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)

    # Probabilities for ROC
    try:
        y_proba = clf.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = np.zeros(len(y_test))

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    try:
        roc_auc = roc_auc_score(y_test, y_proba)
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_test, y_pred_test).tolist()
    from sklearn.metrics import classification_report

# Generate sklearn-style formatted report
    raw_report = classification_report(y_test, y_pred_test)

    # Add model name header + divider
    cr = (
        f"                {model_name.capitalize()} Classification Report:\n"
        f"\t\t-----------------------------------\n"
        + raw_report
    )


    # Churn rate & summary
    churn_rate = float(y.mean())
    if churn_rate > 0.35:
        summary = (
            "⚠ HIGH CHURN RISK: A large fraction of customers are leaving. "
            "Immediate retention actions are recommended."
        )
    elif churn_rate > 0.20:
        summary = (
            "⚠ MODERATE CHURN: Noticeable churn. Focus on at-risk groups such as "
            "month-to-month contracts and high-usage customers."
        )
    else:
        summary = (
            "✔ LOW CHURN: Customer base is relatively stable. Keep monitoring churn "
            "drivers and maintain service quality."
        )

    # Save to visualization cache
    viz_cache["model"] = clf
    viz_cache["X_test"] = X_test
    viz_cache["y_test"] = y_test
    viz_cache["y_pred"] = y_pred_test
    viz_cache["y_proba"] = y_proba
    viz_cache["df_clean"] = df

    mean_proba = float(np.mean(y_proba))

    return train_acc, test_acc, roc_auc, mean_proba, cm, cr, summary


# ----------------------------------------------------
# PREDICT ENDPOINT
# ----------------------------------------------------

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Form(...),
    target_column: str = Form("Churn"),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file.")

    try:
        content = await file.read()
        df = pd.read_csv(StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {e}")

    try:
        (
            train_acc,
            test_acc,
            roc_auc,
            mean_proba,
            cm,
            cr,
            summary,
        ) = train_and_evaluate(df, target_column, model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PredictionResponse(
        model_used=model,
        training_accuracy=round(train_acc, 4),
        testing_accuracy=round(test_acc, 4),
        roc_auc=round(roc_auc, 4),
        mean_prediction_probability=round(mean_proba, 4),
        confusion_matrix=cm,
        classification_report=cr,
        business_summary=summary,
    )


# ----------------------------------------------------
# VISUALIZATION ENDPOINTS
# ----------------------------------------------------

@app.get("/visualize/roc")
def visualize_roc():
    if viz_cache["y_proba"] is None:
        raise HTTPException(400, "Run /predict at least once before visualizing.")

    y_test = viz_cache["y_test"]
    proba = viz_cache["y_proba"]

    fpr, tpr, _ = roc_curve(y_test, proba)

    plt.figure(figsize=(6, 5))
    sns.set_style("whitegrid")
    plt.plot(fpr, tpr, label="ROC Curve", linewidth=2.5)
    plt.plot([0, 1], [0, 1], "r--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    return {"image": fig_to_base64()}


@app.get("/visualize/confusion_heatmap")
def visualize_confusion_heatmap():
    if viz_cache["y_pred"] is None:
        raise HTTPException(400, "Run /predict at least once before visualizing.")

    cm = confusion_matrix(viz_cache["y_test"], viz_cache["y_pred"])

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Heatmap")

    return {"image": fig_to_base64()}


@app.get("/visualize/correlation_heatmap")
def visualize_correlation_heatmap():
    df = viz_cache["df_clean"]
    if df is None:
        raise HTTPException(400, "Run /predict at least once before visualizing.")

    # Only numeric columns for correlation
    corr = df.select_dtypes(include=["int64", "float64"]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")

    return {"image": fig_to_base64()}


@app.get("/visualize/churn_distribution")
def visualize_churn_distribution():
    df = viz_cache["df_clean"]
    if df is None:
        raise HTTPException(400, "Run /predict at least once before visualizing.")

    if "Churn" not in df.columns:
        raise HTTPException(400, "Churn column not found in cleaned data.")

    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x="Churn", palette="viridis")

    plt.title("Churn Distribution")
    plt.xlabel("Churn (0 = No, 1 = Yes)")
    plt.ylabel("Count")

    return {"image": fig_to_base64()}


@app.get("/visualize/feature_importance")
def visualize_feature_importance():
    model = viz_cache["model"]
    df = viz_cache["df_clean"]

    # No model yet
    if model is None:
        return {"image": None, "message": "Run prediction first."}

    base_model = model.named_steps["model"]

    # If model does NOT support feature importance
    if not hasattr(base_model, "feature_importances_"):
        return {
            "image": None,
            "message": (
                "Feature importance is only available for tree-based models:\n\n"
                "- Decision Tree\n"
                "- Random Forest\n"
                "- XGBoost\n"
                "- LightGBM"
            )
        }

    # Tree-based: generate chart
    pre = model.named_steps["preprocessor"]
    num_cols = pre.transformers_[0][2]
    cat_cols = pre.transformers_[1][2]
    ohe = pre.named_transformers_["cat"].named_steps["encode"]

    feature_names = list(num_cols) + list(ohe.get_feature_names_out(cat_cols))
    importances = base_model.feature_importances_

    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).head(15)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=fi, x="importance", y="feature", palette="mako")
    plt.title("Top 15 Important Features")

    return {"image": fig_to_base64(), "message": None}

import shap


@app.get("/explain/customer_count")
def explain_customer_count():
    X_test = viz_cache["X_test"]
    if X_test is None:
        return {"count": 0}
    return {"count": len(X_test)}
@app.get("/explain/shap")
def explain_shap():
    model = viz_cache["model"]
    X_test = viz_cache["X_test"]

    if model is None:
        return {"image": None, "message": "Run prediction first."}

    pre = model.named_steps["preprocessor"]
    final_model = model.named_steps["model"]
    X_trans = pre.transform(X_test)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_trans)
    except:
        explainer = shap.KernelExplainer(final_model.predict_proba, X_trans[:50])
        shap_values = explainer.shap_values(X_trans)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values[1] if isinstance(shap_values, list) else shap_values,
        X_trans,
        show=False
    )

    return {"image": fig_to_base64(), "message": None}
@app.get("/explain/shap_waterfall")
def explain_shap_waterfall(index: int = 0):
    import numpy as np

    model = viz_cache["model"]
    X_test = viz_cache["X_test"]

    if model is None:
        return {"image": None, "message": "Run prediction first."}

    pre = model.named_steps["preprocessor"]
    final_model = model.named_steps["model"]
    X_trans = pre.transform(X_test)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][index]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[index]
            base_val = explainer.expected_value
    except:
        explainer = shap.KernelExplainer(final_model.predict_proba, X_trans[:50])
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][index]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[index]
            base_val = explainer.expected_value

        if isinstance(shap_vals, (float, np.floating, int)):
            shap_vals = np.array([shap_vals])

    plt.figure(figsize=(10, 6))

    shap.waterfall_plot(
        shap.Explanation(values=shap_vals, base_values=base_val, data=X_trans[index]),
        show=False
    )

    return {"image": fig_to_base64(), "message": None}
@app.get("/explain/shap_decision")
def explain_shap_decision(index: int = 0):
    import numpy as np

    model = viz_cache["model"]
    X_test = viz_cache["X_test"]

    if model is None:
        return {"image": None, "message": "Run prediction first."}

    pre = model.named_steps["preprocessor"]
    final_model = model.named_steps["model"]
    X_trans = pre.transform(X_test)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][index]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[index]
            base_val = explainer.expected_value

    except:
        explainer = shap.KernelExplainer(final_model.predict_proba, X_trans[:50])
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][index]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[index]
            base_val = explainer.expected_value

        if isinstance(shap_vals, (float, np.floating, int)):
            shap_vals = np.array([shap_vals])

    plt.figure(figsize=(10, 6))
    shap.decision_plot(base_val, shap_vals, show=False)

    return {"image": fig_to_base64(), "message": None}
@app.get("/explain/shap_customer")
def explain_single_customer(index: int = 0):
    import numpy as np

    model = viz_cache["model"]
    X_test = viz_cache["X_test"]

    if model is None:
        return {"image": None, "message": "Run prediction first."}

    pre = model.named_steps["preprocessor"]
    final_model = model.named_steps["model"]
    X_trans = pre.transform(X_test)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][index]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[index]
            base_val = explainer.expected_value
    except:
        explainer = shap.KernelExplainer(final_model.predict_proba, X_trans[:50])
        shap_values = explainer.shap_values(X_trans)

        if isinstance(shap_values, list):
            shap_vals = shap_values[1][index]
            base_val = explainer.expected_value[1]
        else:
            shap_vals = shap_values[index]
            base_val = explainer.expected_value

        if isinstance(shap_vals, (float, np.floating, int)):
            shap_vals = np.array([shap_vals])

    shap.force_plot(base_val, shap_vals, matplotlib=True, show=False)

    return {"image": fig_to_base64(), "message": None}
@app.get("/explain/shap_text")
def explain_shap_text(index: int = 0):
    import numpy as np

    model = viz_cache["model"]
    X_test = viz_cache["X_test"]

    if model is None:
        return {"text": "Run prediction first."}

    pre = model.named_steps["preprocessor"]
    final_model = model.named_steps["model"]
    X_trans = pre.transform(X_test)

    if hasattr(X_trans, "toarray"):
        X_trans = X_trans.toarray()

    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X_trans)

        shap_vals = (
            shap_values[1][index] if isinstance(shap_values, list)
            else shap_values[index]
        )
    except:
        explainer = shap.KernelExplainer(final_model.predict_proba, X_trans[:50])
        shap_values = explainer.shap_values(X_trans)

        shap_vals = (
            shap_values[1][index] if isinstance(shap_values, list)
            else shap_values[index]
        )

        if isinstance(shap_vals, (float, np.floating, int)):
            shap_vals = np.array([shap_vals])

    feature_contrib = sorted(
        list(enumerate(shap_vals)),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:10]

    response = "Top Reasons for This Customer’s Churn Prediction:\n\n"
    for idx, val in feature_contrib:
        sign = "+" if val >= 0 else "-"
        response += f"{sign}{abs(val):.4f} → Feature Index {idx}\n"

    return {"text": response}
