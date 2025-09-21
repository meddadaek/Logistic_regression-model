import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------
# 1. Load Data
# ------------------------
df = pd.read_csv("iris.csv")

st.title("🌸 Iris Flower Classifier Dashboard")
st.write("This app lets you explore the Iris dataset, train a model, and make predictions.")

# Show dataset
if st.checkbox("Show raw data"):
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    st.write(df['species'].value_counts())

# ------------------------
# 2. Data Exploration
# ------------------------
st.subheader("📊 Data Exploration")

if st.checkbox("Show pairplot"):
    fig = sns.pairplot(df, hue="species")
    st.pyplot(fig)

if st.checkbox("Show feature histograms"):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    sns.histplot(df["sepal_length"], bins=15, ax=axes[0,0])
    sns.histplot(df["sepal_width"], bins=15, ax=axes[0,1])
    sns.histplot(df["petal_length"], bins=15, ax=axes[1,0])
    sns.histplot(df["petal_width"], bins=15, ax=axes[1,1])
    plt.tight_layout()
    st.pyplot(fig)

# ------------------------
# 3. Prepare Features
# ------------------------
X = df.drop("species", axis=1)
y = df["species"]

# ------------------------
# 4. Train/Test Split
# ------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------
# 5. Train Logistic Regression
# ------------------------
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ------------------------
# 6. Evaluation
# ------------------------
st.subheader("⚙️ Model Evaluation")

accuracy = accuracy_score(y_test, y_pred)
st.write("**Accuracy:**", round(accuracy, 3))

st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_, ax=ax)
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# ------------------------
# 7. Prediction
# ------------------------
st.subheader("🌸 Predict a New Flower")

sl = st.slider("Sepal Length", 4.0, 8.0, 5.0)
sw = st.slider("Sepal Width", 2.0, 4.5, 3.0)
pl = st.slider("Petal Length", 1.0, 7.0, 4.0)
pw = st.slider("Petal Width", 0.1, 2.5, 1.0)

if st.button("Predict Species"):
    input_data = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(input_data)
    st.success(f"The predicted species is: **{prediction[0]}**")
