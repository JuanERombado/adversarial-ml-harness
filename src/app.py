# src/app.py
from flask import Flask, jsonify, request
import subprocess, json, os, sys
from metrics import collect

app = Flask(__name__)

PY = sys.executable  # current venv python

def run(cmd):
    completed = subprocess.run([PY] + cmd, capture_output=True, text=True)
    return completed.stdout + completed.stderr

@app.get("/train")
def train():
    out = run(["src/train.py"])
    return jsonify({"status": "trained", "log": out})

@app.get("/attack")
def attack():
    atk = request.args.get("type", "fgsm")
    out = run(["src/adv_gen.py", "--attack", atk])
    return jsonify({"status": f"{atk} done", "log": out})

@app.get("/metrics")
def metrics():
    return jsonify(collect())

@app.get("/report")
def report():
    data = collect()
    html = "<h2>Adversarial ML Report</h2><pre>" + json.dumps(data, indent=2) + "</pre>"
    return html

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)))
