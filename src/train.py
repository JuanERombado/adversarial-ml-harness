# src/train.py
import joblib, json
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

DATA = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    DATA.data, DATA.target, test_size=0.3, random_state=42, stratify=DATA.target
)

model = LogisticRegression(max_iter=200, n_jobs=-1)
model.fit(X_train, y_train)

Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/model.pkl")

acc = accuracy_score(y_test, model.predict(X_test))
(Path("results")).mkdir(exist_ok=True)
(Path("results") / "clean_metrics.json").write_text(json.dumps({"clean_accuracy": acc}, indent=2))

print(f"Model trained. Clean accuracy: {acc:.3f}")
