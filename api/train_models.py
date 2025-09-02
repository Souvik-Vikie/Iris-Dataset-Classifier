# train_models.py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import joblib
from pathlib import Path

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

X, y = load_iris(return_X_y=True)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X, y)
joblib.dump(dt, models_dir / "DecisionTree.pkl")

# KNN
knn = KNeighborsClassifier()
knn.fit(X, y)
joblib.dump(knn, models_dir / "KNN.pkl")

# Logistic Regression
lr = LogisticRegression(max_iter=200)
lr.fit(X, y)
joblib.dump(lr, models_dir / "logistic_regression_model.pkl")

print("✅ Models retrained and saved in models/")
