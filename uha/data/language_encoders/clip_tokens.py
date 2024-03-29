from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn

# !!!OPTIONAL DEPENDENCY "dev" NEEDED!!!
class TokenClip(nn.Module):
    
    def __init__(self, device, pretrained_model= "openai/clip-vit-base-patch32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pretrained_model = pretrained_model
        self.text_encoder = CLIPTextModel.from_pretrained(self.pretrained_model).to(device)
        self.tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.device = device

    def forward(self, batch_text):

        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = True, truncation = True, max_length = 128).to(self.device)
        batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        return batch_text_embed
