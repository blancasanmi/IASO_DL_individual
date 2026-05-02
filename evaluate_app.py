import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import app_
app = app_.app
image = app_.image
volume = app_.volume
VOLUME_PATH = app_.VOLUME_PATH
TENSORS_PATH = app_.TENSORS_PATH

@app.function(
    image=image,
    gpu="L40S",
    timeout=3600,
    volumes={VOLUME_PATH: volume},
)
def final_evaluation(best_m0_config, best_m1_config, best_m2_config):
    import os
    from models import ModelResNetLinear, Model2ResNetMLP, train_pytorch_with_history, eval_pytorch
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import pickle
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend


    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    print("Loading tensors from volume...")
    train_data = torch.load(os.path.join(TENSORS_PATH, "train.pt"))
    val_data   = torch.load(os.path.join(TENSORS_PATH, "val.pt"))
    test_data  = torch.load(os.path.join(TENSORS_PATH, "test.pt"))

    with open(os.path.join(TENSORS_PATH, "label_to_idx.pkl"), "rb") as f:
        label_to_idx = pickle.load(f)
    NUM_CLASSES = len(label_to_idx)
    print(f"Loaded — train: {len(train_data['images'])} | val: {len(val_data['images'])} | classes: {NUM_CLASSES}")

    # ── Build loaders from tensors (no image reading!) ──
    train_loader = DataLoader(
        TensorDataset(train_data["images"], train_data["labels"]),
        batch_size=128, shuffle=True, pin_memory=True, num_workers=2
    )
    val_loader = DataLoader(
        TensorDataset(val_data["images"], val_data["labels"]),
        batch_size=128, shuffle=False, pin_memory=True, num_workers=2
    )
    test_loader = DataLoader(
        TensorDataset(test_data["images"], test_data["labels"]),
        batch_size=128, shuffle=False, pin_memory=True, num_workers=2
    )

    # ── Train each model and collect learning curves ──
    all_histories = {}

    # M0 — untrained baseline (no training, just eval)
    print("\n" + "="*50 + "\nModel 0 — untrained baseline\n" + "="*50)
    m0 = ModelResNetLinear(num_classes=NUM_CLASSES, device=DEVICE, **best_m0_config)
    all_histories["M0"] = train_pytorch_with_history(
        m0.model, train_loader, val_loader, best_m0_config["num_epochs"], best_m0_config["lr"], DEVICE, "M0", best_m0_config["weight_decay"], best_m0_config.get("label_smoothing", 0.0)
    )

    # M1 — retrain with best config to get learning curves
    print("\n" + "="*50 + "\nModel 1 — ResNet + Linear\n" + "="*50)
    m1 = ModelResNetLinear(num_classes=NUM_CLASSES, device=DEVICE, **best_m1_config)
    history_m1 = train_pytorch_with_history(
        m1.model, train_loader, val_loader,
        best_m1_config["num_epochs"], best_m1_config["lr"],
        DEVICE, "M1", best_m1_config["weight_decay"],
        best_m1_config.get("label_smoothing", 0.0)
    )
    all_histories["M1"] = history_m1

    # M2 — retrain with best config to get learning curves
    print("\n" + "="*50 + "\nModel 2 — ResNet + MLP\n" + "="*50)
    m2 = Model2ResNetMLP(num_classes=NUM_CLASSES, device=DEVICE, **best_m2_config)
    history_m2 = train_pytorch_with_history(
        m2.model, train_loader, val_loader,
        best_m2_config["num_epochs"], best_m2_config["lr"],
        DEVICE, "M2", best_m2_config["weight_decay"],
        best_m2_config.get("label_smoothing", 0.0)
    )
    all_histories["M2"] = history_m2

    # ── Final evaluation on test set ──
    print("\n" + "="*50 + "\nFinal Test Evaluation\n" + "="*50)
    test_results = {}

    test_results["M0"] = eval_pytorch(m0.model, test_loader, DEVICE, "test")

    m1.model.load_state_dict(history_m1["best_state"])
    test_results["M1"] = eval_pytorch(m1.model, test_loader, DEVICE, "test")

    m2.model.load_state_dict(history_m2["best_state"])
    test_results["M2"] = eval_pytorch(m2.model, test_loader, DEVICE, "test")

        # At the end of final_evaluation, before return:
    # ✅ strip best_state (tensors can't be serialized by Modal)
    serializable_histories = {}
    for name, h in all_histories.items():
        if h is None:
            serializable_histories[name] = None
        else:
            serializable_histories[name] = {
                k: v for k, v in h.items() if k != "best_state"
            }

        # ── Summary table ──
    print("\n" + "="*60)
    print(f"{'Model':<10} {'Bal. Acc':>10} {'Macro F1':>10} {'Precision':>10} {'Recall':>10}")
    print("-"*60)
    for name, metrics in test_results.items():
        print(f"{name:<10} {metrics['balanced_acc']:>10.4f} {metrics['f1_macro']:>10.4f} "
              f"{metrics['precision']:>10.4f} {metrics['recall']:>10.4f}")

    volume.commit()
    return test_results, serializable_histories
