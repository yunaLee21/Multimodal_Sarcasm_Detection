"""Đóng băng encoder, chỉ mở lại N block transformer cuối để fine-tune nhẹ (Phase 4)."""
import torch.nn as nn


def last_blocks(model, n):
    """N block cuối của PhoBERT/RoBERTa, CLIP vision hoặc ViT, kèm LayerNorm sau block cuối nếu có.

    Không dựa vào tên thuộc tính vì transformers đổi tên giữa các bản (vd 5.x bỏ `vision_model` của CLIP):
    - các block = `nn.ModuleList` dài nhất trong model (`encoder.layer` / `encoder.layers`)
    - LayerNorm cuối = các `nn.LayerNorm` khai báo sau danh sách block, nằm ngoài block. CLIP: `post_layernorm`
      (tạo `pooler_output`, bỏ qua `pre_layrnorm` đứng trước); ViT: `layernorm`; RoBERTa: không có."""
    modules = list(model.named_modules())
    lists = [(i, name, m) for i, (name, m) in enumerate(modules) if isinstance(m, nn.ModuleList) and len(m)]
    if not lists:
        raise ValueError(f"không tìm được các block transformer của {type(model).__name__}")
    start, list_name, layers = max(lists, key=lambda item: len(item[2]))
    if not 0 < n <= len(layers):
        raise ValueError(f"unfreeze_last_n phải trong [1, {len(layers)}], nhận được {n}")
    final_norms = [m for name, m in modules[start + 1:]
                   if isinstance(m, nn.LayerNorm) and not name.startswith(list_name + ".")]
    return list(layers[-n:]) + final_norms


def freeze_encoder(model, freeze, unfreeze_last_n=0):
    """Đóng băng `model` nếu `freeze`, rồi mở lại `unfreeze_last_n` block cuối.

    Trả về các block được mở lại, để `train()` của encoder giữ chúng ở train mode (dropout bật)
    trong khi phần còn đóng băng luôn ở eval mode."""
    if not freeze:
        if unfreeze_last_n:
            raise ValueError("unfreeze_last_n chỉ dùng với freeze: true (freeze: false đã fine-tune toàn bộ encoder)")
        return []
    blocks = last_blocks(model, unfreeze_last_n) if unfreeze_last_n else []  # kiểm tra trước khi đổi model
    model.requires_grad_(False)
    for block in blocks:
        block.requires_grad_(True)
    return blocks
