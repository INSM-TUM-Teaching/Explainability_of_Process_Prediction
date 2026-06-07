import pickle
import os

def check_model():
    model_path = "backend/storage/runs/66f0ac75-7d21-4d28-8c21-432ad7205f4f/artifacts/best_model.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    nodes = getattr(model, "_unpruned_nodes")
    for stage, stage_nodes in nodes.items():
        count = 0
        for node in stage_nodes.values():
            if node['name']:
                print(f"Node found: {node['name']}")
                count += 1
            if count > 5:
                break
        if count > 0:
            break

if __name__ == "__main__":
    check_model()
