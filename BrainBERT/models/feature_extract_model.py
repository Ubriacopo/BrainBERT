from BrainBERT.models import register_model
import torch.nn as nn
import torch
from BrainBERT.models.base_model import BaseModel
from BrainBERT.models.transformer_encoder_input import TransformerEncoderInput

@register_model("feature_extract_model")
class FeatureExtractModel(BaseModel):
    def __init__(self):
        super(FeatureExtractModel, self).__init__()

    def forward(self, inputs):
        out = self.linear_out(inputs)
        return out

    def build_model(self, cfg):
        self.cfg = cfg
        self.linear_out = nn.Linear(in_features=cfg.input_dim, out_features=1) #TODO hardcode out_features
        #TODO hardcode in_features
