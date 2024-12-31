from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, AutoProcessor
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf

class EmbedVLM(nn.Module):
    def __init__(self,
                 device: torch.device,
                 freeze_backbone: bool = True,
                 model_name = "microsoft/Florence-2-base",
                 *args,
                 **kwargs
        ) -> None:

        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.text_encoder = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True).to(device)
        self.text_encoder.requires_grad_(not freeze_backbone)
        self.text_encoder.eval() if freeze_backbone else self.text_encoder.train()
        self.device = device
    
    def forward(self, batch_text_ids):
        batch_text_ids.data["input_ids"] = np.squeeze(batch_text_ids.data["input_ids"])
        batch_text_ids.data["attention_mask"] = np.squeeze(batch_text_ids.data["attention_mask"])
        hidden_states = self.text_encoder.language_model.model.encoder(
            **batch_text_ids
        ).last_hidden_state
        return hidden_states.mean(dim=1)




class TokenVLM(nn.Module):
    def __init__(self, model_name="microsoft/Florence-2-large", *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Initialize the tokenizer - all special tokens are included automatically
        self.tokenizer = AutoProcessor.from_pretrained(model_name, trust_remote_code=True).tokenizer
        
        # Set padding token if not already set
        if self.tokenizer.pad_token is None:
            print('setting padding and eos token')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        

    def forward(self, batch_text):
        # print('start tokenizing'
        # print("inside toenizer")
        # print(batch_text.shape)
       #  batch_text = [str(text.numpy()) if isinstance(text, tf.Tensor) else text for text in batch_text]

        batch_text_ids = self.tokenizer(
            batch_text, 
            return_tensors='pt',
            padding="max_length",
            truncation=True,
            max_length=77,
            return_attention_mask=True
        )
        # decoded = self.tokenizer.batch_decode(batch_text_ids["input_ids"], skip_special_tokens=True)
        # print("Original text:", batch_text)
        # print("Decoded text:", decoded)
        # print('end tokenizing')
        # Remove extra dimension to match expected format
        batch_text_ids.data["input_ids"] = batch_text_ids.data["input_ids"].squeeze(1)
        batch_text_ids.data["attention_mask"] = batch_text_ids.data["attention_mask"].squeeze(1)
        # print(batch_text_ids.data["input_ids"].shape)
        # print('returning tokenized text')
        return batch_text_ids