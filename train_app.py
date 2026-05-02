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
def run_tuning():
    import os, random, pickle
    from models import ModelResNetLinear, Model2ResNetMLP
    import numpy as np
    import torch
    from torch.utils.data import TensorDataset, DataLoader

    def set_seeds(seed=42):
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
    set_seeds()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # ── Load preprocessed tensors ──
    print("Loading tensors from volume...")
    train_data = torch.load(os.path.join(TENSORS_PATH, "train.pt"))
    val_data   = torch.load(os.path.join(TENSORS_PATH, "val.pt"))

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

    # these are the hyperparameters I want to tune for each model (can be different per model if needed)
        
    def sample_config_m0():
        return {
        "lr":              random.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "weight_decay":    random.choice([0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "num_epochs":      random.choice([10, 20, 30]),
        }


    def sample_config_m1():
        return {
        "lr":              random.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
        "weight_decay":    random.choice([0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "num_epochs":      random.choice([10, 20, 30]),
        }

    def sample_config_m2():
        return {
            "lr":              random.choice([1e-4, 5e-4, 1e-3, 5e-3, 1e-2]),
            "weight_decay":    random.choice([0, 1e-5, 1e-4, 1e-3, 1e-2]),
            "num_epochs":      random.choice([10, 20, 30]),
            "hidden_dims":     random.choice([[128], [256], [512], [512, 256], [256, 128], [512, 256, 128]]),
            "dropout":         random.choice([0.1, 0.2, 0.3, 0.4, 0.5]),
        }

    def tune_model(model_class, sample_fn, train_loader, val_loader,
                   num_classes, device, n_iter, label="M?", freeze_backbone=False):
        best_score, best_cfg, results = 0, None, []
        for i in range(n_iter):
            cfg = sample_fn()
            print(f"\n[{label}] iter {i+1}/{n_iter} | config: {cfg}")
            # Only pass freeze_backbone if model_class is ModelResNetLinear
            if model_class == ModelResNetLinear:
                m = model_class(num_classes=num_classes, device=device, freeze_backbone=freeze_backbone, **cfg)
            else:
                m = model_class(num_classes=num_classes, device=device, **cfg)
            m.fit(train_loader, val_loader)
            
            
            val_metrics = m.metrics["val"]
            results.append({
            **cfg,
            "val_bal_acc":  val_metrics["balanced_acc"],
            "val_f1_macro": val_metrics["f1_macro"],
            "val_precision": val_metrics["precision"],
            "val_recall":   val_metrics["recall"],
            })

            if val_metrics["balanced_acc"] > best_score:
                best_score, best_cfg = val_metrics["balanced_acc"], cfg
                torch.save(m.model.state_dict(), os.path.join(VOLUME_PATH, f"best_{label}.pt"))
                        
        volume.commit()
        print(f"\nBest {label} → {best_cfg} | val bal acc={best_score:.4f}")
        return best_cfg, results
    
    print("\n" + "=" * 50 + "\nTuning Model 0\n" + "=" * 50 )
    bestm0, res_m0 = tune_model(ModelResNetLinear, sample_config_m0, train_loader=train_loader, val_loader=val_loader,
                                num_classes=NUM_CLASSES, device=DEVICE, n_iter=10, label="M0", freeze_backbone=True)

    print("=" * 50 + "\nTuning Model 1\n" + "=" * 50)
    best_m1, res_m1 = tune_model(ModelResNetLinear, sample_config_m1,
                                  train_loader, val_loader, NUM_CLASSES, DEVICE, n_iter=10, label="M1", freeze_backbone=False)
    

    print("=" * 50 + "\nTuning Model 2\n" + "=" * 50)
    best_m2, res_m2 = tune_model(Model2ResNetMLP, sample_config_m2,
                                  train_loader, val_loader, NUM_CLASSES, DEVICE, n_iter=10, label="M2")

    return {
    "best_m0": bestm0,
    "res_m0":  res_m0,
    "best_m1": best_m1, 
    "res_m1":  res_m1,
    "best_m2": best_m2, 
    "res_m2":  res_m2,
}

