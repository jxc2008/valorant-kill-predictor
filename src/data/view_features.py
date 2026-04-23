import numpy as np
import os

def inspect_features(filepath):
    print(f"--- Inspecting: {os.path.basename(filepath)} ---")
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}\n")
        return

    # Allow pickling just in case feature_names is an object array
    data = np.load(filepath, allow_pickle=True)
    
    if "feature_names" in data:
        features = data["feature_names"]
        print(f"Columns ({len(features)} total):")
        for i, col in enumerate(features):
            print(f"  {i}: {col}")
    else:
        print("No 'feature_names' array found in this npz file.")
        
    if "X_train" in data:
        X_train = data["X_train"]
        print(f"\nShape of X_train: {X_train.shape}")
        print(f"First 10 rows of X_train:")
        np.set_printoptions(precision=4, suppress=True, linewidth=150)
        print(X_train[:10])
    else:
        print("\nNo 'X_train' array found in this npz file.")
        
    if "X_test" in data:
        X_test = data["X_test"]
        print(f"\nShape of X_test: {X_test.shape}")
    else:
        print("\nNo 'X_test' array found in this npz file.")
        
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # Resolve the paths relative to this script
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, "data")
    
    inspect_features(os.path.join(data_dir, "features.npz"))
    inspect_features(os.path.join(data_dir, "features_with_cluster.npz"))