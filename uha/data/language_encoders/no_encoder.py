import torch.nn as nn

class NoEncoder(nn.Module):
    
    def __init__(self, model_name="", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name

    def forward(self, batch_text):
        return batch_text