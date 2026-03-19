import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoClassifier(nn.Module):
    def __init__(self, model_repo: str, model_name: str, num_classes: int):
        super().__init__()

        self.backbone = torch.hub.load(model_repo, model_name)
        embed_dim = self.backbone.embed_dim

        # Freeze all layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze last 2 transformer blocks
        for name, param in self.backbone.named_parameters():
            if "blocks.10" in name or "blocks.11" in name:
                param.requires_grad = True

        self.embedding_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features["x_norm_clstoken"]
        embeddings = self.embedding_head(cls_token)
        embeddings = F.normalize(embeddings, dim=1)
        logits = self.classifier(embeddings)
        return embeddings, logits