import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎓 Student Pass Predictor", layout="wide", initial_sidebar_state="expanded")

st.title("🎓 Student Performance Prediction App")
st.markdown("Upload a student marks dataset to predict pass/fail outcomes. Customize your threshold and explore model insights.")

# Sidebar for interaction
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("🔁 Pass Mark Threshold", 0, 100, 40)
show_conf_matrix = st.sidebar.checkbox("📉 Show Confusion Matrix", value=True)
show_feature_importance = st.sidebar.checkbox("📊 Show Feature Importance", value=True)

# File Upload
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Check for valid columns
    if df.shape[1] < 3:
        st.error("Dataset should contain at least 3 columns (Name and subject marks).")
    else:
        # Process data
        subject_cols = df.columns[1:]
        df["Average"] = df[subject_cols].mean(axis=1)
        df["Result"] = np.where(df["Average"] >= threshold, "Pass", "Fail")

        st.subheader("📊 Data Preview")
        st.dataframe(df)

        # Charts
        st.subheader("📈 Pass vs Fail Distribution")
        pie = px.pie(df, names='Result', title='Pass/Fail Count', color='Result',
                     color_discrete_map={"Pass": "green", "Fail": "red"})
        st.plotly_chart(pie, use_container_width=True)

        st.subheader("📘 Subject-wise Average")
        avg_marks = df[subject_cols].mean()
        bar_fig = px.bar(x=avg_marks.index, y=avg_marks.values, labels={'x': 'Subjects', 'y': 'Average'},
                         title="Average Marks per Subject", color=avg_marks.values, color_continuous_scale="Viridis")
        st.plotly_chart(bar_fig, use_container_width=True)

        # ML Model
        X = df[subject_cols]
        y = df["Result"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.success(f"✅ Model Accuracy: {acc * 100:.2f}%")

        # Confusion Matrix
        if show_conf_matrix:
            st.subheader("📉 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred, labels=["Pass", "Fail"])
            fig_cm, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Pass", "Fail"], yticklabels=["Pass", "Fail"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig_cm)

        # Feature Importance
        if show_feature_importance:
            st.subheader("📊 Feature Importance")
            importance = pd.DataFrame({"Subject": subject_cols, "Importance": model.feature_importances_})
            importance = importance.sort_values("Importance", ascending=False)
            fig_imp = px.bar(importance, x="Subject", y="Importance", title="Feature Importance", color="Importance",
                             color_continuous_scale="Teal")
            st.plotly_chart(fig_imp, use_container_width=True)

        # Prediction
        st.subheader("🎯 Predict Student Result")
        with st.form("prediction_form"):
            cols = st.columns(len(subject_cols))
            inputs = [cols[i].number_input(f"{sub} Marks", 0, 100, key=sub) for i, sub in enumerate(subject_cols)]
            submitted = st.form_submit_button("Predict")
            if submitted:
                input_data = np.array([inputs])
                result = model.predict(input_data)[0]
                st.info(f"🎓 Prediction: The student will **{result.upper()}**")

        # Download Results
        st.subheader("⬇️ Download Prediction CSV")
        df_out = df.copy()
        csv = df_out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Result Data", data=csv, file_name="predicted_results.csv", mime="text/csv")

        # Model Summary
        with st.expander("ℹ️ Model Summary & Insights"):
            st.markdown("""
            - **Model Used**: Random Forest Classifier  
            - **Features**: Subject marks  
            - **Target**: Pass/Fail based on average marks  
            - **Threshold**: Customizable average to define passing  
            - **Use Case**: Academic performance tracking and early alerts  
            """)
