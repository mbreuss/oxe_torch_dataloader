from transformers import CLIPTextModel, CLIPTokenizer
import torch
import torch.nn as nn
import numpy as np

# !!!OPTIONAL DEPENDENCY "dev" NEEDED!!!
class EmbedClip(nn.Module):
    
    def __init__(self,
                 device: torch.device,
                 freeze_backbone: bool = True,
                 model_name= "openai/clip-vit-base-patch32",
                 *args,
                 **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.text_encoder = CLIPTextModel.from_pretrained(self.model_name).to(device)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.device = device
    
    def forward(self, batch_text_ids):
        batch_text_ids.data["input_ids"] = np.squeeze(batch_text_ids.data["input_ids"]) # [128, 1, 15], remove unused middle dim
        batch_text_ids.data["attention_mask"] = np.squeeze(batch_text_ids.data["attention_mask"]) # [128, 1, 15], remove unused middle dim
        # batch_text_embed = self.text_encoder(**batch_text_ids).last_hidden_state
        batch_text_embed = self.text_encoder(**batch_text_ids)["pooler_output"]
        return batch_text_embed


class TokenClip(nn.Module):
    def __init__(self, model_name= "openai/clip-vit-base-patch32", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
    
    def forward(self, batch_text):
        batch_text_ids = self.tokenizer(batch_text, return_tensors = 'pt', padding = "max_length", truncation = True, max_length = 77)
        return batch_text_ids # data: [input_ids (shape=[1,15]), attention_mask (shape=[1,15])]
