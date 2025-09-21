# 🌸 Iris Flower Classifier Dashboard

An interactive **Streamlit web app** for exploring the classic **Iris dataset**, training a **Logistic Regression model**, and making predictions.  

This app allows you to:  
- Explore the dataset (pairplots, histograms).  
- Train a machine learning model.  
- Evaluate model performance.  
- Predict species of a new flower with sliders.  

---

## 📂 Project Structure

iris_app.py # main Streamlit app
iris.csv # dataset file
README.md # this file

markdown
Copier le code

---

## ⚙️ Requirements

- Python 3.7+  
- Libraries:  
  - `streamlit`  
  - `pandas`  
  - `numpy`  
  - `seaborn`  
  - `matplotlib`  
  - `scikit-learn`  

Install dependencies with:  

```bash
pip install streamlit pandas numpy seaborn matplotlib scikit-learn
▶️ How to Run
Place iris.csv (Iris dataset) in the same folder as iris_app.py.

If you don’t have it, download from: Iris dataset on Kaggle.

Run the Streamlit app:

bash
Copier le code
streamlit run iris_app.py
A browser window will open with the dashboard.

🎨 Features
1. Dataset Exploration
Toggle to show the raw dataset.

Visualize relationships with a pairplot.

Check feature distributions with histograms.

2. Model Training
Splits data into training and testing sets.

Trains a Logistic Regression model.

3. Model Evaluation
Displays accuracy.

Shows a classification report.

Visualizes a confusion matrix.

4. Prediction
Use sliders to enter sepal and petal measurements.

Click Predict to classify the flower into one of:

setosa

versicolor

virginica

📸 Demo (Conceptual)
less
Copier le code
🌸 Iris Flower Classifier Dashboard
-----------------------------------
📊 Data Exploration
[ ] Show raw data
[ ] Show pairplot
[ ] Show feature histograms

⚙️ Model Evaluation
Accuracy: 0.967
Classification Report: ...

🌸 Predict a New Flower
Sepal Length: [slider]
Sepal Width : [slider]
Petal Length: [slider]
Petal Width : [slider]
[ Predict Species ]
📝 Notes
Streamlit automatically reloads when you save changes.

You can modify the dataset or model for experimentation (try RandomForestClassifier or SVM).

✨ Built with Streamlit and Scikit-learn.
