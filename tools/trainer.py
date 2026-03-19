import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim import Adam, AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from pytorch_metric_learning import losses, miners

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_parser import load_config
from model.backbone import DinoClassifier


class Trainer:
    def __init__(self, config):
        self.cfg = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(self.cfg.result_dir, exist_ok=True)

        self.model = DinoClassifier(
            model_repo=self.cfg.model_repo,
            model_name=self.cfg.model_name,
            num_classes=len(self.cfg.class_names)
        ).to(self.device)

        self._build_dataloaders()
        self._build_losses()
        self._build_optimizer()

        self.log_file = os.path.join(self.cfg.result_dir, "training_log.csv")
        with open(self.log_file, "w") as f:
            f.write("epoch,train_loss,val_loss,train_acc,val_acc\n")

        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def _build_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            # transforms.RandomRotation(5),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        train_dataset = datasets.ImageFolder(self.cfg.train_dir, transform=train_transform)
        val_dataset = datasets.ImageFolder(self.cfg.val_dir, transform=val_transform)

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(val_dataset, batch_size=self.cfg.batch_size, shuffle=False, num_workers=4)

    def _build_losses(self):
        self.ce_loss = nn.CrossEntropyLoss()
        self.miner = miners.MultiSimilarityMiner()

        # hard_triplet_loss
        if self.cfg.use_hard_triplet_loss:
            self.triplet_loss = losses.TripletMarginLoss(margin=0.3)

        # soft_triplet_loss
        elif self.cfg.use_soft_triplet_loss:
            self.triplet_loss = losses.SoftTripleLoss(num_classes=len(self.cfg.class_names), embedding_size=128)
        else:
            self.triplet_loss = None

    def _build_optimizer(self):
        params = filter(lambda p: p.requires_grad, self.model.parameters())
        if self.cfg.optimizer == "adam":
            self.optimizer = Adam(params, lr=self.cfg.learning_rate)
        else:
            self.optimizer = AdamW(params, lr=self.cfg.learning_rate)

    def _compute_loss(self, embeddings, logits, labels):
        ce = self.ce_loss(logits, labels)
        if self.triplet_loss is None:
            return ce, ce.item(), 0.0

        hard_pairs = self.miner(embeddings, labels)
        triplet = self.triplet_loss(embeddings, labels, hard_pairs)
        total = ce + self.cfg.triplet_weight * triplet
        return total, ce.item(), triplet.item()

    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0

        pbar = tqdm(self.train_loader, desc="Train")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            embeddings, logits = self.model(images)
            loss, ce_val, triplet_val = self._compute_loss(embeddings, logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (torch.argmax(logits, dim=1) == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", ce=f"{ce_val:.4f}", triplet=f"{triplet_val:.4f}")

        return total_loss / len(self.train_loader), correct / total

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0

        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Val"):
                images, labels = images.to(self.device), labels.to(self.device)
                embeddings, logits = self.model(images)
                loss, _, _ = self._compute_loss(embeddings, logits, labels)

                total_loss += loss.item()
                correct += (torch.argmax(logits, dim=1) == labels).sum().item()
                total += labels.size(0)

        return total_loss / len(self.val_loader), correct / total

    def train(self):
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, self.cfg.epochs + 1):
            print(f"\nEpoch {epoch}/{self.cfg.epochs}")
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

            for key, val in zip(["train_loss", "val_loss", "train_acc", "val_acc"],
                                 [train_loss, val_loss, train_acc, val_acc]):
                self.history[key].append(val)

            with open(self.log_file, "a") as f:
                f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f},{train_acc:.4f},{val_acc:.4f}\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), os.path.join(self.cfg.result_dir, "best_model.pth"))
                print("Best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= self.cfg.patience:
                    print("Early stopping triggered.")
                    break

        self._plot_curves()

    def _plot_curves(self):
        epochs = range(1, len(self.history["train_loss"]) + 1)

        plt.figure()
        plt.plot(epochs, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs, self.history["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.result_dir, "loss_curve.png"))
        plt.close()

        plt.figure()
        plt.plot(epochs, self.history["train_acc"], label="Train Acc")
        plt.plot(epochs, self.history["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.cfg.result_dir, "accuracy_curve.png"))
        plt.close()

        print("Training curves saved.")


if __name__ == "__main__":
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    cfg = load_config(config_path)
    trainer = Trainer(cfg)
    trainer.train()