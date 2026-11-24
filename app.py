import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

st.set_page_config(
    page_title="NEXUS Smart Business Demand Predictor",
    page_icon="📊",
    layout="wide",
)

DEFAULT_DATA_PATH = (
    r"C:\Users\Krupa\Downloads\NEXUS_Smart_Business_Analysis_200000_Records.csv"
)
DEFAULT_MODEL_PATH = "NEXUS_Trained_Model.pkl"
DROP_COLS = [
    "record_id",
    "store_id",
    "date",
    "next_month_sales_units",
    "sales_growth_pct",
]


@st.cache_data(show_spinner="Loading and preprocessing dataset...")
def load_and_preprocess(csv_source):
    df = pd.read_csv(csv_source)
    df.columns = df.columns.str.strip()
    df = df.drop_duplicates()

    raw_df = df.copy()
    raw_df["parsed_date"] = pd.to_datetime(
        raw_df["date"], format="%d-%m-%Y", errors="coerce"
    )

    proc_df = df.copy()
    proc_df["date"] = pd.to_datetime(
        proc_df["date"], format="%d-%m-%Y", errors="coerce"
    )
    proc_df["month"] = proc_df["date"].dt.month
    proc_df["year"] = proc_df["date"].dt.year

    proc_df = proc_df.dropna(subset=["demand_level"])
    existing_drops = [c for c in DROP_COLS if c in proc_df.columns]
    proc_df = proc_df.drop(columns=existing_drops)

    return raw_df, proc_df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    num_process = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_process = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer([
        ("num", num_process, num_cols),
        ("cat", cat_process, cat_cols),
    ])

    clf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


@st.cache_resource(show_spinner="Training Random Forest model...")
def train_model_cached(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int):
    pipe = build_pipeline(X_train)
    pipe.named_steps["classifier"].set_params(n_estimators=n_estimators)
    pipe.fit(X_train, y_train)
    return pipe


@st.cache_resource(show_spinner="Loading saved model...")
def load_model_cached(model_path: str):
    return joblib.load(model_path)


st.title("📊 NEXUS Smart Business — Demand Level Prediction")
st.caption(
    "Predict `demand_level` (Low / Medium / High) and explore sales trends & model performance."
)

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("Data & Model Settings")
    uploaded_csv = st.file_uploader("Upload CSV Dataset (optional)", type=["csv"])
    csv_path = st.text_input("Or Local CSV Path", value=DEFAULT_DATA_PATH)
    model_path = st.text_input("Saved Model (.pkl) Path", value=DEFAULT_MODEL_PATH)
    n_estimators = st.slider("Random Forest Trees (n_estimators)", 50, 200, 100, 25)
    test_size = st.slider("Test Split (%)", 10, 40, 20, 5) / 100.0

data_source = uploaded_csv if uploaded_csv is not None else csv_path

try:
    raw_df, proc_df = load_and_preprocess(data_source)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

if "demand_level" not in proc_df.columns:
    st.error("Target column `demand_level` is missing from the dataset.")
    st.stop()

X = proc_df.drop(columns=["demand_level"])
y = proc_df["demand_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

model = None
col_m1, col_m2 = st.sidebar.columns(2)
with col_m1:
    train_clicked = st.button("Train Model", type="primary", use_container_width=True)
with col_m2:
    load_clicked = st.button("Load .pkl", use_container_width=True)

if train_clicked:
    model = train_model_cached(X_train, y_train, n_estimators)
    joblib.dump(model, model_path)
    st.sidebar.success(f"Trained & saved to `{model_path}`")
elif load_clicked:
    try:
        model = load_model_cached(model_path)
        st.sidebar.success(f"Loaded `{model_path}`")
    except Exception as e:
        st.sidebar.error(f"Could not load model: {e}")
else:
    # Auto-load if model file already exists, otherwise train once
    try:
        model = load_model_cached(model_path)
    except Exception:
        model = train_model_cached(X_train, y_train, n_estimators)

# --- Tabs ---
tab_overview, tab_eval, tab_predict, tab_trends = st.tabs([
    "📁 Dataset Overview",
    "📈 Model Evaluation",
    "🔮 Live Prediction",
    "🗓️ Monthly Sales Trends",
])

# --- Tab 1: Overview ---
with tab_overview:
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Rows", f"{len(raw_df):,}")
    k2.metric("Raw Columns", raw_df.shape[1])
    k3.metric("Missing Values", int(raw_df.isnull().sum().sum()))
    k4.metric("Duplicate Rows", int(raw_df.duplicated().sum()))

    st.subheader("First 10 Records")
    st.dataframe(raw_df.head(10), use_container_width=True)

    st.subheader("Target Distribution (`demand_level`)")
    st.bar_chart(y.value_counts())

# --- Tab 2: Evaluation ---
with tab_eval:
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Precision (Weighted)", f"{prec:.4f}")
    m3.metric("Recall (Weighted)", f"{rec:.4f}")
    m4.metric("F1 Score (Weighted)", f"{f1:.4f}")

    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_,
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    with c2:
        st.subheader("Top 10 Feature Importances")
        feat_names = model.named_steps["preprocessor"].get_feature_names_out()
        importances = model.named_steps["classifier"].feature_importances_
        top_imp = (
            pd.Series(importances, index=feat_names)
            .sort_values(ascending=False)
            .head(10)
            .sort_values(ascending=True)
        )
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        top_imp.plot(kind="barh", ax=ax2)
        ax2.set_xlabel("Importance")
        ax2.set_ylabel("Feature")
        st.pyplot(fig2)
        plt.close(fig2)

    st.subheader("Classification Report")
    rep_df = pd.DataFrame(
        classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ).transpose()
    st.dataframe(rep_df, use_container_width=True)

    st.subheader("Sample Actual vs Predicted (First 25 Test Rows)")
    comp_df = pd.DataFrame({
        "Actual": y_test.values[:25],
        "Predicted": y_pred[:25],
    })
    st.dataframe(comp_df, use_container_width=True)

# --- Tab 3: Live Prediction ---
with tab_predict:
    st.subheader("Single Record Prediction")
    sample_row = X.median(numeric_only=True).to_dict()
    cat_columns = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    input_data = {}
    cols = st.columns(3)

    for idx, col in enumerate(cat_columns):
        options = sorted(X[col].dropna().astype(str).unique().tolist())
        input_data[col] = cols[idx % 3].selectbox(col, options=options)

    num_columns = [c for c in X.columns if c not in cat_columns]
    for idx, col in enumerate(num_columns):
        default_val = float(sample_row.get(col, 0.0))
        if pd.api.types.is_integer_dtype(X[col]):
            input_data[col] = cols[(idx + len(cat_columns)) % 3].number_input(
                col, value=int(default_val), step=1
            )
        else:
            input_data[col] = cols[(idx + len(cat_columns)) % 3].number_input(
                col, value=float(default_val), format="%.2f"
            )

    if st.button("Predict Demand Level", type="primary"):
        input_df = pd.DataFrame([input_data])[X.columns]
        pred_label = model.predict(input_df)[0]
        pred_proba = model.predict_proba(input_df)[0]

        st.success(f"Predicted Demand Level: **{pred_label}**")
        prob_df = pd.DataFrame({
            "Demand Level": model.classes_,
            "Probability": np.round(pred_proba * 100, 2),
        })
        st.dataframe(prob_df, use_container_width=True)

    st.divider()
    st.subheader("Batch CSV Prediction")
    batch_file = st.file_uploader(
        "Upload CSV for batch demand_level prediction", type=["csv"], key="batch_csv"
    )
    if batch_file is not None:
        batch_raw = pd.read_csv(batch_file)
        batch_prep = batch_raw.copy()
        batch_prep.columns = batch_prep.columns.str.strip()
        if "date" in batch_prep.columns:
            dt = pd.to_datetime(
                batch_prep["date"], format="%d-%m-%Y", errors="coerce"
            )
            batch_prep["month"] = dt.dt.month
            batch_prep["year"] = dt.dt.year
        for col in X.columns:
            if col not in batch_prep.columns:
                batch_prep[col] = np.nan
        batch_preds = model.predict(batch_prep[X.columns])
        batch_raw["predicted_demand_level"] = batch_preds
        st.dataframe(batch_raw.head(20), use_container_width=True)
        st.download_button(
            "Download Predictions CSV",
            data=batch_raw.to_csv(index=False).encode("utf-8"),
            file_name="NEXUS_Demand_Predictions.csv",
            mime="text/csv",
        )

# --- Tab 4: Monthly Sales Trends ---
with tab_trends:
    st.subheader("Monthly Sales Trend by Product Category")
    if {"parsed_date", "product_category", "units_sold"}.issubset(raw_df.columns):
        trend_df = raw_df.dropna(subset=["parsed_date"]).copy()
        trend_df["year_month"] = (
            trend_df["parsed_date"].dt.to_period("M").astype(str)
        )

        categories = sorted(
            trend_df["product_category"].dropna().unique().tolist()
        )
        selected_cats = st.multiselect(
            "Select Product Categories",
            options=categories,
            default=categories[:4] if len(categories) > 4 else categories,
        )
        metric_col = st.selectbox(
            "Sales Metric",
            options=[
                c
                for c in ["units_sold", "revenue_inr", "profit_inr"]
                if c in trend_df.columns
            ],
        )

        if selected_cats:
            filtered = trend_df[trend_df["product_category"].isin(selected_cats)]
            pivot_trend = (
                filtered.groupby(["year_month", "product_category"])[metric_col]
                .sum()
                .unstack(fill_value=0)
                .sort_index()
            )
            st.line_chart(pivot_trend)
            st.dataframe(pivot_trend.tail(12), use_container_width=True)
        else:
            st.info("Select at least one product category to display the trend.")
    else:
        st.warning(
            "Required columns (`date`, `product_category`, `units_sold`) not found."
        )