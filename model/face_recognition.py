from backbones import get_model as get_backbone_model
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_face_recognition_model(cfg):
    return FaceRecognition(cfg)


class FaceRecognition(nn.Module):
    def __init__(self, cfg):
        super(FaceRecognition, self).__init__()
        self.cfg = cfg
        self.backbone = get_backbone_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
        self.fc_1 = nn.Linear(cfg.embedding_size, cfg.embedding_size)
        self.fc_2 = nn.Linear(cfg.embedding_size, cfg.num_classes + 1)

    def forward(self, x):
        feat = self.backbone(x)
        feat = F.relu(self.fc_1(feat))
        score = self.fc_2(feat)
        return score

    def load_backbone_weight(self):
        dict_checkpoint = torch.load(self.cfg.backbone_weight)
        backbone_weight = dict_checkpoint
        self.backbone.load_state_dict(backbone_weight)
