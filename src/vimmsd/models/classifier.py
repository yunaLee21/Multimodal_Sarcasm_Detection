import torch
import torch.nn as nn

from vimmsd.models.fusion import FUSIONS, manifold_mixup
from vimmsd.models.heads import HEAD_TYPES, CascadeHead, HierarchicalHead, LabelIndex, mlp
from vimmsd.models.image_encoder import ImageEncoder
from vimmsd.models.text_encoder import TextEncoder

MODALITY_NEEDS = {
    "text": (True, False),
    "text_aux": (True, False),
    "image": (False, True),
    "concat": (True, True),
    "cross_attn": (True, True),
    "multistream": (True, False),
    "multistream_clip": (True, True),
}


class SarcasmClassifier(nn.Module):
    """Lắp ghép encoder + fusion + head phân loại.

    Tách `encode` và `classify` để phần analysis (Grad-CAM, tráo ảnh) có thể can thiệp
    vào feature ở giữa mà không cần chạy lại encoder.

    `head_cfg` (model.head trong config, xem models/heads.py):
        type: flat | hierarchical | cascade
        axis_heads: thêm head T/I (text / ảnh đứng riêng có mỉa mai không), cho loss phụ λ_axis;
            với hierarchical, xác suất T/I (không truyền gradient ngược) là đầu vào thêm của tầng C
        rationale_dim: > 0 thì thêm phép chiếu vector fusion sang không gian embedding reasoning (loss λ_rat)
    `forward` trả về logits 4 lớp (với hierarchical/cascade là log xác suất), `forward_outputs` trả thêm
    các đầu ra phụ cho loss."""

    def __init__(self, modality, text_encoder, image_encoder, fusion_cfg, num_classes, head_hidden, dropout,
                 head_cfg=None, labels=None):
        super().__init__()
        head_cfg = dict(head_cfg or {})
        self.head_type = head_cfg.get("type", "flat")
        if self.head_type not in HEAD_TYPES:
            raise ValueError(f"model.head.type phải thuộc {HEAD_TYPES}, nhận được {self.head_type!r}")
        self.modality = modality
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.fusion = FUSIONS[modality](
            text_dim=text_encoder.hidden_size if text_encoder else None,
            image_dim=image_encoder.hidden_size if image_encoder else None,
            dropout=dropout,
            **fusion_cfg,
        )
        in_dim = self.fusion.out_dim
        use_axis = bool(head_cfg.get("axis_heads", False))
        self.axis_head = mlp(in_dim, head_hidden, 2, dropout) if use_axis else None
        if self.head_type == "flat":
            self.head = mlp(in_dim, head_hidden, num_classes, dropout)  # tên tham số head.* như bản cũ
        else:
            idx = LabelIndex(labels)
            if self.head_type == "hierarchical":
                self.head = HierarchicalHead(in_dim, head_hidden, dropout, idx, use_axis=use_axis)
            else:
                self.head = CascadeHead(in_dim, head_hidden, dropout, idx, head_cfg.get("cascade_threshold", 0.5))
        rationale_dim = int(head_cfg.get("rationale_dim") or 0)
        self.rationale_proj = nn.Linear(in_dim, rationale_dim) if rationale_dim > 0 else None
        self.prior_shift = None  # dict tensor, đặt bằng set_prior_shift; không lưu vào checkpoint

    @property
    def tokenizer(self):
        return self.text_encoder.tokenizer if self.text_encoder else None

    @property
    def image_processor(self):
        return self.image_encoder.image_processor if self.image_encoder else None

    def encode(self, batch):
        text = image = None
        if self.text_encoder is not None:
            text = self.text_encoder(batch["input_ids"], batch["attention_mask"])
            if "ocr_input_ids" in batch:
                text["ocr"] = self.text_encoder(batch["ocr_input_ids"], batch["ocr_attention_mask"])["pooled"]
                text["ocr_present"] = batch["ocr_present"]
                text["description"] = self.text_encoder(
                    batch["description_input_ids"], batch["description_attention_mask"])["pooled"]
                text["description_present"] = batch["description_present"]
        if self.image_encoder is not None:
            image = self.image_encoder(batch["pixel_values"])
        return text, image

    def set_prior_shift(self, shift):
        """Bật (dict từ heads.compute_prior_shift) hoặc tắt (None) hiệu chỉnh prior lúc suy luận."""
        self.prior_shift = None if shift is None else {k: v.detach().clone() for k, v in shift.items()}

    def _shift(self, device):
        return None if self.prior_shift is None else {k: v.to(device) for k, v in self.prior_shift.items()}

    def heads(self, z):
        out = {"z": z}
        shift = self._shift(z.device)
        axis_logits = self.axis_head(z) if self.axis_head is not None else None
        if axis_logits is not None:
            out["axis_logits"] = axis_logits
        if self.head_type == "flat":
            logits = self.head(z)
            out["logits"] = logits + shift["logits"].to(logits.dtype) if shift is not None else logits
        elif self.head_type == "hierarchical":
            axis_probs = torch.sigmoid(axis_logits.float()).detach() if axis_logits is not None else None
            out.update(self.head(z, axis_probs, shift))
        else:
            out.update(self.head(z, shift))
        if self.rationale_proj is not None:
            out["rationale"] = self.rationale_proj(z)
        return out

    def classify(self, text, image):
        return self.heads(self.fusion(text, image))["logits"]

    def forward_outputs(self, batch):
        return self.heads(self.fusion(*self.encode(batch)))

    def forward_mixup(self, batch, alpha, minority_labels=()):
        z = self.fusion(*self.encode(batch))
        mixed, lam, perm = manifold_mixup(z, batch["labels"], alpha, minority_labels)
        out = self.heads(mixed)
        out["mixup_lambda"], out["mixup_permutation"] = lam, perm
        return out

    def forward(self, batch):
        return self.forward_outputs(batch)["logits"]

    def param_groups(self, lr_encoder, lr_head, weight_decay):
        encoder_params, head_params = [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            is_encoder = name.startswith(("text_encoder.", "image_encoder."))
            (encoder_params if is_encoder else head_params).append(p)
        groups = [{"params": head_params, "lr": lr_head, "weight_decay": weight_decay}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay})
        return groups


def build_model(cfg):
    m = cfg.model
    if m.modality not in MODALITY_NEEDS:
        raise ValueError(f"model.modality phải thuộc {list(MODALITY_NEEDS)}, nhận được {m.modality!r}")
    needs_text, needs_image = MODALITY_NEEDS[m.modality]
    text_encoder = TextEncoder(m.text_encoder.name, m.text_encoder.freeze,
                               m.text_encoder.get("unfreeze_last_n", 0)) if needs_text else None
    image_encoder = ImageEncoder(m.image_encoder.name, m.image_encoder.freeze,
                                 m.image_encoder.get("unfreeze_last_n", 0)) if needs_image else None
    return SarcasmClassifier(
        modality=m.modality,
        text_encoder=text_encoder,
        image_encoder=image_encoder,
        fusion_cfg=dict(m.get("fusion", {})),
        num_classes=len(cfg.data.labels),
        head_hidden=m.head_hidden,
        dropout=m.dropout,
        head_cfg=dict(m.get("head") or {}),
        labels=list(cfg.data.labels),
    )
