import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

shap_dir = "/Users/iliapanayotov/Projects/Explainability_of_Process_Prediction/backend/storage/runs/435de4b6-97fa-42de-b288-18e0e93486cc/artifacts/shap"
# Read the summary json if it exists
import json
try:
    with open(shap_dir + "/../summary.json", 'r') as f:
        print(json.load(f))
except Exception as e:
    pass
