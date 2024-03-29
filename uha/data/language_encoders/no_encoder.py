import torch.nn as nn

class NoEncoder(nn.Module):
    
    def __init__(self, device=None, pretrained_model="", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pretrained_model = pretrained_model
        self.device = device

    def forward(self, batch_text):
        return batch_text