import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("IRIS.csv")
df = data.copy()

encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])

features = df.drop("species", axis=1)
target = df["species"]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

X_train, X_test, y_train, y_test = train_test_split(
    scaled_features, target, test_size=0.2, random_state=42, stratify=target
)

model_options = {
    "Logistic Regression": LogisticRegression(max_iter=250),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42)
}

performance = {}
for key, clf in model_options.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    performance[key] = accuracy_score(y_test, preds)

top_model_name = max(performance, key=performance.get)
final_model = model_options[top_model_name]
joblib.dump(final_model, "iris_best_model.pkl")

def classify_flower(a, b, c, d):
    sample_df = pd.DataFrame([[a, b, c, d]], columns=features.columns)
    sample_scaled = scaler.transform(sample_df)
    prediction = final_model.predict(sample_scaled)
    return encoder.inverse_transform(prediction)[0]


st.title("🌸 Iris Flower Classifier")
st.write("An interactive model to identify Iris species using petal and sepal dimensions.")

a = st.slider("Sepal Length (cm)", float(features['sepal_length'].min()), float(features['sepal_length'].max()), 5.0)
b = st.slider("Sepal Width (cm)", float(features['sepal_width'].min()), float(features['sepal_width'].max()), 3.2)
c = st.slider("Petal Length (cm)", float(features['petal_length'].min()), float(features['petal_length'].max()), 1.4)
d = st.slider("Petal Width (cm)", float(features['petal_width'].min()), float(features['petal_width'].max()), 0.2)

if st.button("Predict Species"):
    species = classify_flower(a, b, c, d)
    st.success(f"Predicted Iris species: **{species}** 🌼")

st.subheader("📊 Model Accuracy")
st.bar_chart(pd.DataFrame(performance.items(), columns=["Model", "Accuracy"]).set_index("Model"))

st.subheader("🌿 Species Distribution")
fig1, ax1 = plt.subplots()
df_species = data["species"].value_counts()
ax1.pie(df_species, labels=df_species.index, autopct="%1.1f%%", startangle=90)
ax1.axis("equal")
st.pyplot(fig1)

st.subheader("🌼 Feature Scatter Plot")
col_x, col_y = st.columns(2)
with col_x:
    x_feat = st.selectbox("X-axis", features.columns)
with col_y:
    y_feat = st.selectbox("Y-axis", features.columns, index=2)

fig2, ax2 = plt.subplots()
sns.scatterplot(data=data, x=x_feat, y=y_feat, hue="species", ax=ax2)
st.pyplot(fig2)
