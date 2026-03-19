import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms

from models.backbone import DinoClassifier


class DinoInference:

    def __init__(self, model_path, class_names, device=None):

        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        self.class_names = class_names

        self.model = DinoClassifier(
            num_classes=len(class_names)
        )

        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225]
            )
        ])


    def predict(self, image):

        # cv2 image is BGR → convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():

            _, logits = self.model(tensor)

            probs = F.softmax(logits, dim=1)

            confidence, pred = torch.max(probs, dim=1)

        class_name = self.class_names[pred.item()]
        confidence = confidence.item()

        return class_name, confidence