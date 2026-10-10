"""Path D: auxiliary coarse-level classification loss for Mask2Former.

The class head keeps its K fine logits (+ no-object). A coarse logit is the
log-sum-exp of the fine logits in that group, i.e. the log of the summed
fine probabilities, so a query that confuses ``vehicle fallback`` with
``truck`` is still rewarded for "large vehicle". The coarse CE is applied
to matched queries only; unmatched queries are already fully supervised
toward no-object by the fine CE.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from mask2former.modeling.criterion import SetCriterion


class HierarchicalSetCriterion(SetCriterion):
    def loss_labels(self, outputs, targets, indices, num_masks):
        losses = super().loss_labels(outputs, targets, indices, num_masks)
        logits = outputs["pred_logits"].float()
        idx = self._get_src_permutation_idx(indices)
        if idx[0].numel() == 0:
            losses["loss_ce_coarse"] = logits.sum() * 0.0
            return losses
        fine = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        coarse_logits = torch.logsumexp(logits[idx].unsqueeze(-1) + self.coarse_log_mask, dim=1)
        losses["loss_ce_coarse"] = F.cross_entropy(coarse_logits, self.coarse_of_fine[fine])
        return losses


def make_hierarchical(criterion: SetCriterion, coarse_of_fine: list[int], loss_weight: float) -> None:
    """Turn a built SetCriterion into a HierarchicalSetCriterion in place.

    ``coarse_of_fine[c]`` is the coarse group of contiguous class ``c``; the
    no-object class gets its own extra group.
    """
    n_groups = max(coarse_of_fine) + 1
    groups = torch.tensor(coarse_of_fine + [n_groups])
    mask = F.one_hot(groups, n_groups + 1).float()
    criterion.__class__ = HierarchicalSetCriterion
    criterion.register_buffer("coarse_of_fine", groups.to(criterion.empty_weight.device))
    criterion.register_buffer("coarse_log_mask", mask.log().to(criterion.empty_weight.device))
    wd = criterion.weight_dict
    for k in [k for k in wd if k == "loss_ce" or k.startswith("loss_ce_")]:
        wd["loss_ce_coarse" + k[len("loss_ce"):]] = wd[k] * loss_weight
