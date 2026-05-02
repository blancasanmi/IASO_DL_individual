import modal 

app = modal.App("plant-pathology")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(["libsm6", "libxext6", "libxrender-dev", "build-essential"])  # For matplotlib headless
    .pip_install([
        "torch==2.0.1", "torchvision==0.15.2",
        "scikit-learn", "pandas", "numpy<2",
        "Pillow", "matplotlib",
    ])
)

volume = modal.Volume.from_name("plant-pathology-data", create_if_missing=True)
VOLUME_PATH = "/data"
TENSORS_PATH = "/data/tensors"


@app.local_entrypoint()
def main():
    import json
    import pandas as pd
    
    print("\n" + "="*60)
    print("Step 1: Preprocessing data...")
    print("="*60)
    prepare_data.remote()
    
    print("\n" + "="*60)
    print("Step 2: Tuning models...")
    print("="*60)
    result = run_tuning.remote()
    
    # Save tuning results
    pd.DataFrame(result["res_m0"]).to_csv("results/" \
    "res_m0.csv", index=False)
    pd.DataFrame(result["res_m1"]).to_csv("results/" \
    "res_m1.csv", index=False)
    pd.DataFrame(result["res_m2"]).to_csv("results/" \
    "res_m2.csv", index=False)
    
    with open("results/best_configs.json", "w") as f:
        json.dump({
            "M0": result["best_m0"], 
            "M1": result["best_m1"], 
            "M2": result["best_m2"]
        }, f, indent=2)
    
    print("\nBest M0:", result["best_m0"])
    print("Best M1:", result["best_m1"])
    print("Best M2:", result["best_m2"])
    
    print("\n" + "="*60)
    print("Step 3: Final evaluation...")
    print("="*60)
    test_results, histories = final_evaluation.remote(
        result["best_m0"], 
        result["best_m1"], 
        result["best_m2"]
    )
    
    # Save final results
    pd.DataFrame(histories).to_csv("results/learning_curves.csv", index=True)
    pd.DataFrame(test_results).to_csv("results/test_results.csv", index=True)
    
    print("\nFinal test results:", test_results)
    print("Learning curves saved to results/learning_curves.csv")
    print("Test results saved to results/test_results.csv")
