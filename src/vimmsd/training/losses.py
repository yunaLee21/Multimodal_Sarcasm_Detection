import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(counts, mode, beta=0.999):
    counts = torch.tensor(counts, dtype=torch.float).clamp(min=1)
    if mode == "inverse":
        w = 1.0 / counts
    elif mode == "sqrt_inverse":
        w = 1.0 / counts.sqrt()
    elif mode == "effective":
        # Cui et al. 2019, "Class-Balanced Loss Based on Effective Number of Samples"
        w = (1.0 - beta) / (1.0 - torch.pow(beta, counts))
    else:
        raise ValueError(f"class_weight không hợp lệ: {mode}")
    return w / w.sum() * len(counts)  # chuẩn hóa để trung bình weight = 1


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)  (Lin et al. 2017).

    (1 - p_t)^gamma giảm trọng số các mẫu đã phân loại dễ (p_t cao), để gradient tập trung
    vào mẫu khó — thường là mẫu của lớp hiếm. alpha là class weight (có thể None)."""

    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha if alpha is not None else None)

    def forward(self, logits, targets):
        log_p = F.log_softmax(logits.float(), dim=-1)
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)
        loss = -((1 - log_pt.exp()) ** self.gamma) * log_pt
        if self.alpha is not None:
            loss = loss * self.alpha[targets]
        return loss.mean()


class SarcasmLoss(nn.Module):
    """Loss train dùng đầu ra đầy đủ của SarcasmClassifier.forward_outputs (docs/experiment_design.md mục 5).

    flat / hierarchical: `base` (CE / focal) trên logits 4 lớp; với hierarchical logits là log xác suất nên đây
        đúng là NLL. Thêm λ_S · BCE(tầng S, nhãn != not) cho hierarchical.
    cascade: BCE(head M, nhãn == multi) trên mọi mẫu + CE(head R) chỉ trên mẫu không phải multi.
    λ_axis · BCE(head T/I): axis_targets="gold" dùng nhãn suy từ nhãn gốc (châm biếm có liên quan tới text / ảnh);
        "teacher" dùng phán đoán "text / ảnh đứng riêng" của teacher. Mẫu không có nhãn của teacher: lớp not / text /
        image thì hai nghĩa trùng nhau nên dùng nhãn suy từ nhãn gốc, lớp multi thì bỏ (chỉ teacher biết thuộc
        trường hợp 1-4 nào).
    λ_rat · (1 - cos(phép chiếu vector fusion, embedding reasoning)) trên mẫu có reasoning đạt rubric.
    Trả về (loss, dict các thành phần) để log riêng từng phần."""

    uses_outputs = True

    def __init__(self, base, head_type, labels, lambda_s=0.0, lambda_axis=0.0, axis_targets="gold",
                 lambda_rat=0.0, route_weight=None):
        super().__init__()
        from vimmsd.models.heads import LabelIndex

        if axis_targets not in ("gold", "teacher"):
            raise ValueError(f"model.head.axis_targets phải là gold | teacher, nhận được {axis_targets!r}")
        self.base = base
        self.head_type = head_type
        self.idx = LabelIndex(labels)
        self.lambda_s, self.lambda_axis, self.lambda_rat = float(lambda_s), float(lambda_axis), float(lambda_rat)
        self.axis_targets = axis_targets
        self.register_buffer("route_weight", route_weight if route_weight is not None else None)
        remap = torch.full((4,), -1, dtype=torch.long)
        for k, label in enumerate(self.idx.non_multi):
            remap[label] = k
        self.register_buffer("route_remap", remap)

    def _axis(self, batch, y):
        gold = self.idx.axis_targets(y)
        if self.axis_targets == "gold":
            return gold, torch.ones_like(y, dtype=torch.bool)
        target, mask = gold, y != self.idx.multi
        if "axis_teacher" in batch:
            has = batch["axis_teacher_mask"].bool()
            target = torch.where(has.unsqueeze(-1), batch["axis_teacher"].float(), gold)
            mask = mask | has
        return target, mask

    def forward(self, out, batch):
        y = batch["labels"]
        parts = {}
        if self.head_type == "cascade":
            parts["multi"] = F.binary_cross_entropy_with_logits(out["m_logit"].float(), (y == self.idx.multi).float())
            keep = y != self.idx.multi
            if keep.any():
                parts["route"] = F.cross_entropy(out["r_logits"][keep].float(), self.route_remap[y[keep]],
                                                 weight=self.route_weight)
        else:
            parts["cls"] = self.base(out["logits"].float(), y)
            if self.head_type == "hierarchical" and self.lambda_s:
                parts["s"] = self.lambda_s * F.binary_cross_entropy_with_logits(
                    out["s_logit"].float(), (y != self.idx.not_).float())
        if self.lambda_axis and "axis_logits" in out:
            target, mask = self._axis(batch, y)
            if mask.any():
                parts["axis"] = self.lambda_axis * F.binary_cross_entropy_with_logits(
                    out["axis_logits"][mask].float(), target[mask])
        if self.lambda_rat and "rationale" in out and "rationale_mask" in batch:
            mask = batch["rationale_mask"].bool()
            if mask.any():
                cos = F.cosine_similarity(out["rationale"][mask].float(), batch["rationale_emb"][mask].float(), dim=-1)
                parts["rat"] = self.lambda_rat * (1 - cos).mean()
        return sum(parts.values()), {k: float(v.detach()) for k, v in parts.items()}


def build_training_loss(cfg, class_counts):
    """Loss cho Trainer: SarcasmLoss bọc build_loss. Head flat không bật loss phụ nào thì kết quả giống hệt build_loss."""
    head = cfg.model.get("head") or {}
    head_type = head.get("type", "flat")
    route_weight = None
    if head_type == "cascade" and cfg.loss.get("class_weight"):
        from vimmsd.models.heads import LabelIndex

        idx = LabelIndex(cfg.data.labels)
        route_weight = compute_class_weights([class_counts[i] for i in idx.non_multi], cfg.loss.class_weight,
                                             cfg.loss.get("beta", 0.999))
    return SarcasmLoss(build_loss(cfg.loss, class_counts), head_type, cfg.data.labels,
                       lambda_s=cfg.loss.get("lambda_s", 0.0), lambda_axis=cfg.loss.get("lambda_axis", 0.0),
                       axis_targets=head.get("axis_targets", "gold"), lambda_rat=cfg.loss.get("lambda_rat", 0.0),
                       route_weight=route_weight)


def build_loss(loss_cfg, class_counts):
    mode = loss_cfg.get("class_weight")
    weight = compute_class_weights(class_counts, mode, loss_cfg.get("beta", 0.999)) if mode else None
    if loss_cfg.name == "ce":
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=loss_cfg.get("label_smoothing", 0.0))
    if loss_cfg.name == "focal":
        return FocalLoss(gamma=loss_cfg.get("gamma", 2.0), alpha=weight)
    raise ValueError(f"loss không hợp lệ: {loss_cfg.name}")
