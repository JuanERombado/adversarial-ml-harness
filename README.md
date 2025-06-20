# Adversarial-ML Harness

Tiny Flask service that **trains a scikit-learn model, generates FGSM / PGD adversarial examples with IBM ART, and reports robustness metrics** — useful as a demo of AI application-security testing.

<br>

## ✨ What it shows
| Skill | How it’s demonstrated |
|-------|-----------------------|
| ML fundamentals | Trains a logistic-regression classifier on the Iris dataset |
| Adversarial testing | Uses Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) to attack the model |
| Secure SDLC mindset | CI workflow in GitHub Actions, `.gitignore`, isolated Python venv |
| DevOps / APIs | Exposes train / attack / metrics endpoints via Flask for automation or dashboards |

<br>

## Quick start

```bash
git clone https://github.com/JuanERombado/adversarial-ml-harness.git
cd adversarial-ml-harness
python -m venv venv
source venv/Scripts/activate   # Linux/mac: source venv/bin/activate
pip install -r requirements.txt

# Train + attack once
python src/train.py
python src/adv_gen.py --attack fgsm
python src/adv_gen.py --attack pgd
python src/metrics.py
