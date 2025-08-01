import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Live Student Dashboard", layout="wide", page_icon="📘")
st.title("📘 Live Student Pass/Fail Prediction Dashboard")

# Load dataset
df = pd.read_csv("Student_Mark.csv")
subject_cols = ['Math', 'Science', 'English', 'History', 'Computer']

# Sidebar - Input Panel
st.sidebar.header("📋 Enter Student Marks")
math = st.sidebar.slider("Math", 0, 100, 60)
science = st.sidebar.slider("Science", 0, 100, 60)
english = st.sidebar.slider("English", 0, 100, 60)
history = st.sidebar.slider("History", 0, 100, 60)
computer = st.sidebar.slider("Computer", 0, 100, 60)

pass_logic = st.sidebar.radio("Pass Criteria", ["Average ≥ 60", "At least 3 Subjects ≥ 60"])
chart_type = st.sidebar.selectbox("Chart Type", ["Histogram", "Pie Chart", "Bar Chart", "Box Plot"])
show_table = st.sidebar.checkbox("🔍 Show Full Dataset")

# Optional: Filter dataset by pass/fail
filter_option = st.sidebar.radio("Filter Students", ["All", "Pass Only", "Fail Only"])

# Create input data
data_input = pd.DataFrame([[math, science, english, history, computer]], columns=subject_cols)

# Apply pass/fail logic
if pass_logic == "Average ≥ 60":
    df["Average"] = df[subject_cols].mean(axis=1)
    df["Pass"] = df["Average"].apply(lambda x: 1 if x >= 60 else 0)
    input_avg = data_input.mean(axis=1)[0]
    input_pass = 1 if input_avg >= 60 else 0
    data_input["Average"] = input_avg
    data_input["Pass"] = input_pass
    display_metric = "Average"
else:
    df["Subjects_Passed"] = df[subject_cols].apply(lambda row: sum(row >= 60), axis=1)
    df["Pass"] = df["Subjects_Passed"].apply(lambda x: 1 if x >= 3 else 0)
    input_passed = sum(data_input.iloc[0] >= 60)
    input_pass = 1 if input_passed >= 3 else 0
    data_input["Subjects_Passed"] = input_passed
    data_input["Pass"] = input_pass
    display_metric = "Subjects_Passed"

# Add name for user input row
data_input["Name"] = "🧑 You"
if "Name" not in df.columns:
    df["Name"] = [f"Student {i+1}" for i in range(len(df))]

# Append user input row
df_combined = pd.concat([df, data_input], ignore_index=True)

# Filter view
if filter_option == "Pass Only":
    df_combined = df_combined[df_combined["Pass"] == 1]
elif filter_option == "Fail Only":
    df_combined = df_combined[df_combined["Pass"] == 0]

# Train model
X = df[subject_cols]
y = df["Pass"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
acc = accuracy_score(y_test, model.predict(X_test))

# Layout
col1, col2 = st.columns([2, 1])

# Charts
with col1:
    st.subheader("📊 Live Chart (Includes Your Data)")
    if chart_type == "Histogram":
        fig = px.histogram(df_combined, x=display_metric, color="Pass", nbins=10,
                           title=f"{display_metric} Distribution",
                           color_discrete_map={1: "green", 0: "red"})
    elif chart_type == "Pie Chart":
        fig = px.pie(df_combined, names="Pass", title="Pass/Fail Ratio",
                     color='Pass', hole=0.4,
                     color_discrete_map={1: "green", 0: "red"})
    elif chart_type == "Bar Chart":
        fig = px.bar(df_combined.sort_values(by=display_metric, ascending=False).head(10),
                     x="Name", y=display_metric, color="Pass",
                     title=f"Top 10 by {display_metric}",
                     color_discrete_map={1: "green", 0: "red"})
    else:
        fig = px.box(df_combined, y=display_metric, color="Pass",
                     title=f"Box Plot of {display_metric} by Pass Status",
                     color_discrete_map={1: "green", 0: "red"})

    st.plotly_chart(fig, use_container_width=True)

# Sidebar metrics and user prediction
with col2:
    st.subheader("🧠 Model Info")
    st.metric("Accuracy", f"{acc:.2f}")
    st.metric("Criteria", pass_logic)

    st.subheader("📌 Your Prediction")
    if input_pass == 1:
        st.success("🎉 Based on the input, the student is likely to PASS!")
    else:
        st.error("⚠️ Based on the input, the student may FAIL.")

    if st.button("📂 Save this prediction"):
        st.toast("Saved locally. (Add backend to persist)")

# Full data display
if show_table:
    st.subheader("🔍 Full Dataset View")
    st.dataframe(df_combined.reset_index(drop=True), use_container_width=True)

st.markdown("---")
st.caption("📘 Made with ❤️ by Rohita")
