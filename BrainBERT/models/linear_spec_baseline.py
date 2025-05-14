import torch.nn as nn

from BrainBERT.models import register_model
from BrainBERT.models.base_model import BaseModel


@register_model("linear_spec_baseline")
class LinearSpecModel(BaseModel):
    def __init__(self):
        super(LinearSpecModel, self).__init__()

    def forward(self, inputs):
        out = self.linear_out(inputs)
        return out

    def build_model(self, cfg, input_dim):
        self.cfg = cfg
        self.linear_out = nn.Linear(in_features=input_dim, out_features=1) #TODO hardcode out_features
        #TODO hardcode in_features
