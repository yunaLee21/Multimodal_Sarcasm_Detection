import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from vimmsd.models.freezing import freeze_encoder


class TextEncoder(nn.Module):
    """Wrapper PhoBERT: trả về hidden state từng token (cho cross-attention) và vector <s> (cho concat)."""

    def __init__(self, name="vinai/phobert-base", freeze=False, unfreeze_last_n=0):
        super().__init__()
        self.model = AutoModel.from_pretrained(name)
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.hidden_size = self.model.config.hidden_size
        self.frozen = freeze
        self.unfrozen_blocks = freeze_encoder(self.model, freeze, unfreeze_last_n)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()  # phần đóng băng luôn tắt dropout
            for block in self.unfrozen_blocks:
                block.train(mode)
        return self

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        tokens = out.last_hidden_state
        return {"tokens": tokens, "mask": attention_mask.bool(), "pooled": tokens[:, 0]}
