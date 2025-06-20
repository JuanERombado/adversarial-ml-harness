# src/metrics.py
import json, glob
from pathlib import Path

def collect():
    result = {}
    for j in glob.glob("results/*_metrics.json"):
        with open(j) as f:
            result.update({Path(j).stem: json.load(f)})
    return result

if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
