from typing import Dict

import torch
import torch.nn as nn

from game_model import BallStateType
from vision.ball_classifier.model import SmallModel, LABELS
from vision.ball_classifier.train import MODEL_PATH


class BallTypeClassifier:
    model: nn.Module

    def __init__(self):
        self.model = SmallModel.load(MODEL_PATH)

    def predict_type(self, img: torch.Tensor) -> BallStateType:
        img = img.unsqueeze(dim=0)
        outputs = self.model(img)
        _, argmax_cls_tensor = torch.max(outputs, 1)
        class_index = int(argmax_cls_tensor[0])
        return LABELS[class_index]

    def predict_probabilities(self, img: torch.Tensor) -> Dict[BallStateType, float]:
        img = img.unsqueeze(dim=0)
        outputs = self.model(img)
        softmax_outputs = torch.nn.functional.softmax(outputs)

        return {
            label: float(softmax_outputs[0, label_idx])
            for label_idx, label in enumerate(LABELS)
        }
