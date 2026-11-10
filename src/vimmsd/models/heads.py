"""Head phân loại cho CoT distillation (docs/experiment_design.md mục 5).

- flat: head MLP 4 lớp như cũ (giữ nguyên tên tham số `head.*` để load được checkpoint cũ).
- hierarchical (M2): tầng S "có châm biếm không" (sigmoid) × tầng C "loại nào" (softmax text / image / multi),
  nhân xác suất mềm, train end-to-end. Trả về log P(4 lớp) thay cho logits: softmax(log p) = p, nên CrossEntropy
  trên đầu ra này đúng bằng NLL, predict()/argmax và class weight dùng lại được không cần sửa.
- cascade: tái lập kiểu pipeline phân cấp của Vu et al. (2025) trong một model: head M (multi hay không, train trên
  mọi mẫu) và head R (not / text / image, chỉ train trên mẫu không phải multi), lúc suy luận định tuyến cứng.

Head phụ T/I ("text / ảnh đứng riêng có mỉa mai không") và phép chiếu căn chỉnh reasoning nằm ở SarcasmClassifier,
dùng chung cho mọi loại head."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HEAD_TYPES = ("flat", "hierarchical", "cascade")
LABEL_NAMES = {"not": "not-sarcasm", "text": "text-sarcasm", "image": "image-sarcasm", "multi": "multi-sarcasm"}


def mlp(in_dim, hidden, out_dim, dropout):
    """Cùng dạng head cũ: Dropout → Linear → GELU → Dropout → Linear."""
    return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(hidden, out_dim))


class LabelIndex:
    """Vị trí của 4 nhãn trong cfg.data.labels, để head không phụ thuộc thứ tự nhãn trong config."""

    def __init__(self, labels):
        labels = list(labels)
        missing = [v for v in LABEL_NAMES.values() if v not in labels]
        if missing or len(labels) != 4:
            raise ValueError(f"head phân cấp cần đúng 4 nhãn {list(LABEL_NAMES.values())}, nhận được {labels}")
        self.not_ = labels.index(LABEL_NAMES["not"])
        self.text = labels.index(LABEL_NAMES["text"])
        self.image = labels.index(LABEL_NAMES["image"])
        self.multi = labels.index(LABEL_NAMES["multi"])
        self.sarcastic = [self.text, self.image, self.multi]   # thứ tự lớp của tầng C (hierarchical)
        self.non_multi = [self.not_, self.text, self.image]    # thứ tự lớp của head R (cascade)

    def axis_targets(self, y):
        """Nhãn phụ suy từ nhãn gốc: châm biếm có liên quan tới text / ảnh (involves_text, involves_image)."""
        t = (y == self.text) | (y == self.multi)
        i = (y == self.image) | (y == self.multi)
        return torch.stack([t, i], dim=-1).float()


class HierarchicalHead(nn.Module):
    def __init__(self, in_dim, hidden, dropout, idx, use_axis=False):
        super().__init__()
        self.idx = idx
        self.use_axis = use_axis
        self.s = mlp(in_dim, hidden, 1, dropout)
        self.c = mlp(in_dim + (2 if use_axis else 0), hidden, 3, dropout)

    def forward(self, z, axis_probs=None, shift=None):
        s_logit = self.s(z).squeeze(-1)
        c_in = torch.cat([z, axis_probs.to(z.dtype)], dim=-1) if self.use_axis else z
        c_logits = self.c(c_in)
        if shift is not None:
            s_logit = s_logit + shift["s"].to(s_logit.dtype)
            c_logits = c_logits + shift["c"].to(c_logits.dtype)
        log_s = F.logsigmoid(s_logit.float())
        log_c = F.log_softmax(c_logits.float(), dim=-1)
        cols = [None] * 4
        cols[self.idx.not_] = F.logsigmoid(-s_logit.float())
        for k, label in enumerate(self.idx.sarcastic):
            cols[label] = log_s + log_c[:, k]
        return {"logits": torch.stack(cols, dim=-1), "s_logit": s_logit, "c_logits": c_logits}


class CascadeHead(nn.Module):
    """Định tuyến cứng: P(multi) >= threshold thì kết luận multi, ngược lại lấy argmax của head R.
    Xác suất 4 lớp trả về chỉ để argmax đúng luật định tuyến (dùng cho predict), không phải xác suất hiệu chỉnh."""

    def __init__(self, in_dim, hidden, dropout, idx, threshold=0.5):
        super().__init__()
        self.idx = idx
        self.threshold = threshold
        self.m = mlp(in_dim, hidden, 1, dropout)
        self.r = mlp(in_dim, hidden, 3, dropout)

    def forward(self, z, shift=None):
        m_logit = self.m(z).squeeze(-1)
        r_logits = self.r(z)
        if shift is not None:
            m_logit = m_logit + shift["m"].to(m_logit.dtype)
            r_logits = r_logits + shift["r"].to(r_logits.dtype)
        p_m = torch.sigmoid(m_logit.float())
        p_r = torch.softmax(r_logits.float(), dim=-1)
        route = p_m >= self.threshold
        cols = [None] * 4
        # đi nhánh multi: P(multi) = p_m >= 0,5 >= (1 - p_m) * p_r nên argmax luôn là multi;
        # đi nhánh R: P(multi) = 0, các lớp còn lại theo đúng head R
        cols[self.idx.multi] = torch.where(route, p_m, torch.zeros_like(p_m))
        for k, label in enumerate(self.idx.non_multi):
            cols[label] = torch.where(route, (1 - p_m) * p_r[:, k], p_r[:, k])
        probs = torch.stack(cols, dim=-1)
        return {"logits": probs.clamp(min=1e-8).log(), "m_logit": m_logit, "r_logits": r_logits}


def _logit(p):
    return math.log(p) - math.log1p(-p)


def compute_prior_shift(head_type, train_counts, target_counts, labels):
    """Dịch logit để prior của model khớp phân bố đích (prior của val = public test), theo từng tầng của head.

    flat: log p_đích - log p_train cho 4 logit (logit adjustment).
    hierarchical: tầng S dịch logit(P_đích(có châm biếm)) - logit(P_train(...)), tầng C dịch theo prior có điều kiện.
    cascade: head M dịch theo P(multi), head R theo prior có điều kiện trên 3 lớp còn lại.
    Đếm lớp được kẹp tối thiểu 1 (val chỉ có 1 mẫu text-sarcasm)."""
    tr = torch.tensor(train_counts, dtype=torch.float).clamp(min=1)
    tg = torch.tensor(target_counts, dtype=torch.float).clamp(min=1)
    if head_type == "flat":
        return {"logits": (tg / tg.sum()).log() - (tr / tr.sum()).log()}
    idx = LabelIndex(labels)
    if head_type == "hierarchical":
        p_tr = 1 - float(tr[idx.not_] / tr.sum())
        p_tg = 1 - float(tg[idx.not_] / tg.sum())
        c_tr, c_tg = tr[idx.sarcastic], tg[idx.sarcastic]
        return {"s": torch.tensor(_logit(p_tg) - _logit(p_tr)),
                "c": (c_tg / c_tg.sum()).log() - (c_tr / c_tr.sum()).log()}
    if head_type == "cascade":
        p_tr = float(tr[idx.multi] / tr.sum())
        p_tg = float(tg[idx.multi] / tg.sum())
        r_tr, r_tg = tr[idx.non_multi], tg[idx.non_multi]
        return {"m": torch.tensor(_logit(p_tg) - _logit(p_tr)),
                "r": (r_tg / r_tg.sum()).log() - (r_tr / r_tr.sum()).log()}
    raise ValueError(f"head.type không hợp lệ: {head_type}")
