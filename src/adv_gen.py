"""
Generate adversarial samples (FGSM or PGD) for the trained Iris model
and write metrics + numpy arrays to the results directory.

Run examples:
    python src/adv_gen.py --attack fgsm
    python src/adv_gen.py --attack pgd
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import SklearnClassifier


# ---------- helper functions -------------------------------------------------


def to_labels(preds: np.ndarray) -> np.ndarray:
    """IBM ART returns probability vectors; convert to hard labels."""
    return preds.argmax(axis=1) if preds.ndim > 1 else preds


def get_attack(name: str, clf: SklearnClassifier):
    """Return an ART attack object based on the chosen name."""
    if name == "fgsm":
        return FastGradientMethod(estimator=clf, eps=0.2)
    if name == "pgd":
        return ProjectedGradientDescent(estimator=clf, eps=0.3, max_iter=40)
    raise ValueError("attack must be 'fgsm' or 'pgd'")


# ---------- main workflow ----------------------------------------------------


def main(attack_name: str) -> None:
    # load model + wrap with ART
    model = joblib.load("models/model.pkl")
    clf = SklearnClassifier(model=model)

    # load full Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # craft adversarial examples
    attack = get_attack(attack_name, clf)
    X_adv = attack.generate(x=X)

    # predictions and metrics
    clean_preds = to_labels(clf.predict(X))
    adv_preds = to_labels(clf.predict(X_adv))

    clean_acc = accuracy_score(y, clean_preds)
    adv_acc = accuracy_score(y, adv_preds)

    # save outputs
    Path("results").mkdir(exist_ok=True)
    np.save(f"results/{attack_name}_samples.npy", X_adv)
    np.save(f"results/{attack_name}_labels.npy", y)

    metrics = {
        "attack": attack_name,
        "clean_accuracy": float(clean_acc),
        "adv_accuracy": float(adv_acc),
        "attack_success_rate": float(1 - adv_acc / clean_acc),
    }
    (Path("results") / f"{attack_name}_metrics.json").write_text(
        json.dumps(metrics, indent=2)
    )

    print(
        f"{attack_name.upper()} complete — clean_acc={clean_acc:.3f} "
        f"adv_acc={adv_acc:.3f}"
    )


# ---------- CLI entrypoint ---------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attack", choices=["fgsm", "pgd"], required=True, help="Attack type"
    )
    args = parser.parse_args()
    main(args.attack)
