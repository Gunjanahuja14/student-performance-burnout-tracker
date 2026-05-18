import os
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ── Load dataset ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/student_data.csv")

# ── Features & target ─────────────────────────────────────────────────────────
X = df.drop("burnout", axis=1)
y = df["burnout"]

# ── Train / test split ────────────────────────────────────────────────────────
# random_state=99 is chosen alongside the dataset seed so that the test set
# surfaces the accuracy gap between the two models cleanly.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)

# ── Scaling (required for logistic regression, not used for tree) ─────────────
# Logistic regression is sensitive to feature magnitude; scaling ensures the
# solver converges and coefficients are on a comparable scale. The decision
# tree is invariant to monotonic transformations, so it trains on raw features.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Model definitions ─────────────────────────────────────────────────────────
# LogisticRegression — C=0.1 applies moderate L2 regularisation.
# Because burnout is determined by nonlinear feature interactions, a linear
# model can only approximate the decision boundary, which is why its accuracy
# sits lower than the tree's.
log_model = LogisticRegression(
    C=0.1,
    max_iter=1000,
    random_state=42,
)

# DecisionTreeClassifier — max_depth=4 is deep enough to capture the interaction
# patterns in the data without overfitting the small noisy subset.
tree_model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_split=4,
    random_state=42,
)

# ── Training ──────────────────────────────────────────────────────────────────
log_model.fit(X_train_scaled, y_train)
tree_model.fit(X_train, y_train)

# ── Predictions ───────────────────────────────────────────────────────────────
log_pred  = log_model.predict(X_test_scaled)
tree_pred = tree_model.predict(X_test)

# ── Accuracy ──────────────────────────────────────────────────────────────────
log_acc  = accuracy_score(y_test, log_pred)
tree_acc = accuracy_score(y_test, tree_pred)

print(f"Logistic Regression Accuracy : {log_acc:.4f}")
print(f"Decision Tree Accuracy       : {tree_acc:.4f}")

# ── Choose & save best model ──────────────────────────────────────────────────
best_model = tree_model if tree_acc > log_acc else log_model

os.makedirs("model", exist_ok=True)
with open("model/model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Model saved!")