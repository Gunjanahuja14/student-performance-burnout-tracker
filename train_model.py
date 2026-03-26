import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv("data/student_data.csv")

# Features & target
X = df.drop("burnout", axis=1)
y = df["burnout"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
log_model = LogisticRegression()
tree_model = DecisionTreeClassifier()

# Train
log_model.fit(X_train, y_train)
tree_model.fit(X_train, y_train)

# Predict
log_pred = log_model.predict(X_test)
tree_pred = tree_model.predict(X_test)

# Accuracy
log_acc = accuracy_score(y_test, log_pred)
tree_acc = accuracy_score(y_test, tree_pred)

print("Logistic Regression Accuracy:", log_acc)
print("Decision Tree Accuracy:", tree_acc)

# Choose best model
best_model = tree_model if tree_acc > log_acc else log_model

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved!")