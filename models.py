import torch
import torch.nn as nn
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score, classification_report


def get_resnet18(pretrained: bool = True) -> nn.Module:
        """Load ResNet18 with ImageNet weights, remove classification head."""
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Identity()  # output: 512-d feature vector
        return model

def train_pytorch_with_history(model, train_loader, val_loader, num_epochs,
                                lr, device, label, weight_decay, label_smoothing=0.0):
        model = model.to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        scaler = torch.amp.GradScaler('cuda')

        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "best_state": None}
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            # ── Train ──
            model.train()
            train_loss, train_correct, train_total, num_batches = 0.0, 0, 0, 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()
                train_total += labels.size(0)
                num_batches += 1
            scheduler.step()

            # ── Validate ──
            model.eval()
            val_loss, all_preds, all_labels = 0.0, [], []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    with torch.amp.autocast('cuda'):
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    all_preds.append(outputs.argmax(1))
                    all_labels.append(labels)

            y_pred = torch.cat(all_preds).cpu().numpy()
            y_true = torch.cat(all_labels).cpu().numpy()
            val_bal_acc = balanced_accuracy_score(y_true, y_pred)

            # ── Record ──
            history["train_loss"].append(train_loss / num_batches)
            history["val_loss"].append(val_loss / len(val_loader))
            history["train_acc"].append(train_correct / train_total)
            history["val_acc"].append(val_bal_acc)

            if val_bal_acc > best_val_acc:
                best_val_acc = val_bal_acc
                history["best_state"] = {k: v.clone() for k, v in model.state_dict().items()}

            print(f"[{label}] Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_loss/num_batches:.4f} | Val Bal Acc: {val_bal_acc:.4f}")

        return history
    
def evaluate(y_true, y_pred, split: str = "val"):
    """
    Compute and print classification metrics.
    
    Args:
        y_true: True labels (numpy array)
        y_pred: Predicted labels (numpy array)
        split: Name of the split for printing ("train", "val", "test")
    
    Returns:
        Dictionary with balanced_acc, f1_macro, f1_per_class, precision, recall
    """
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    prec_mac = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_mac = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    print(f"\n[{split}] balanced_acc: {bal_acc:.4f} | macro_F1: {f1_mac:.4f} | "
        f"precision: {prec_mac:.4f} | recall: {rec_mac:.4f}")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    return dict(
        balanced_acc=bal_acc,
        f1_macro=f1_mac,
        f1_per_class=f1_per,
        precision=prec_mac,
        recall=rec_mac
    )

def eval_pytorch(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    split: str,
    use_amp: bool = True
    ):
        """
        Evaluate model on a dataloader and compute metrics.
        
        Args:
            model: Trained neural network
            loader: DataLoader to evaluate on
            device: Device ("cuda" or "cpu")
            split: Split name for printing ("train", "val", "test")
            use_amp: Whether to use automatic mixed precision
        
        Returns:
            Dictionary with metrics from evaluate()
        """
        model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    preds = model(images).argmax(1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels)
        
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        
        return evaluate(y_true, y_pred, split)

def train_pytorch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    lr: float,
    device: str,
    label: str,
    weight_decay: float,
    use_amp: bool = True,
):
    """
    Train a PyTorch model with mixed precision, learning rate scheduling, and early stopping.
    
    Args:
        model: Neural network module
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on ("cuda" or "cpu")
        label: Label for printing (e.g., "M1", "M2")
        weight_decay: L2 regularization strength
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        (trained_model, history_dict) where history_dict contains train_loss, val_loss, val_acc
    """
    model = model.to(device)
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_state = None
    
    # For learning curves
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "best_state": None,
    }
    
    print(f"\n{'='*70}")
    print(f"[{label}] {num_epochs} epochs | lr={lr} | weight_decay={weight_decay} | AMP={use_amp}")
    print(f"{'='*70}")
    
    for epoch in range(1, num_epochs + 1):
        # ── Training ──
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = criterion(model(images), labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        scheduler.step()
        
        # ── Validation ──
        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda', enabled=use_amp):
                    logits = model(images)
                    batch_loss = criterion(logits, labels)
                    preds = logits.argmax(1)
                
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())
                val_loss += batch_loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        val_bal_acc = balanced_accuracy_score(y_true, y_pred)
        
        # Store history
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_bal_acc)
        
        # Early stopping
        if val_bal_acc > best_val_acc:
            best_val_acc = val_bal_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            history["best_state"] = best_state
        
        if epoch % 2 == 0 or epoch in (1, num_epochs):
            print(f"[{label}] Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                f"Val Bal Acc: {val_bal_acc:.4f}")
    
    print(f"[{label}] Training complete. Best val balanced acc: {best_val_acc:.4f}\n")
    model.load_state_dict(best_state)
    return model, history            


class MLP(nn.Module):
    """Multi-layer perceptron head with batch norm, ReLU, and dropout."""
    def __init__(self, in_dim: int, num_classes: int, hidden_dims: list[int], dropout: float = 0.3):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResNetModular(nn.Module):
    """ResNet18 backbone + modular head (can be Linear or MLP)."""
    def __init__(self, head: nn.Module, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = get_resnet18(pretrained=True)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            for param in self.backbone.parameters():
                param.requires_grad = True
        
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

# ── Wrapper classes ──
class ModelResNetLinear:
    """ResNet18 backbone + single linear head (= logistic regression)"""
    def __init__(self, num_classes, device, num_epochs=20, lr=1e-4, weight_decay=1e-4, name="M1", freeze_backbone=False):
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.name = name
        self.freeze_backbone = freeze_backbone
        self.model = ResNetModular(nn.Linear(512, num_classes), freeze_backbone=self.freeze_backbone)

    def fit(self, train_loader, val_loader):
        self.model, self.history = train_pytorch(
            self.model, train_loader, val_loader,
            self.num_epochs, self.lr, self.device, self.name, self.weight_decay
        )
        self.metrics = {
            "train": eval_pytorch(self.model, train_loader, self.device, "train"),
            "val":   eval_pytorch(self.model, val_loader,   self.device, "val"),
        }
        return self.metrics

class Model2ResNetMLP:
    """ResNet18 backbone + MLP head"""
    def __init__(self, num_classes, device, hidden_dims=None, dropout=0.3,
                num_epochs=20, lr=1e-4, weight_decay=1e-4):
        self.device = device
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.model = ResNetModular(
            MLP(512, num_classes, hidden_dims or [256], dropout),
            freeze_backbone=False
        )

    def fit(self, train_loader, val_loader):
        self.model, self.history = train_pytorch(
            self.model, train_loader, val_loader,
            self.num_epochs, self.lr, self.device, "M2", self.weight_decay
        )
        self.metrics = {
            "train": eval_pytorch(self.model, train_loader, self.device, "train"),
            "val":   eval_pytorch(self.model, val_loader,   self.device, "val"),
        }
        return self.metrics