"""Model definitions: EVA backbone, multi-view fusion, triple-branch, VAAF."""

from .eva_backbone import (
    EVA_X,
    eva_x_base_patch16,
    load_evax_init_weights,
    load_single_view_weights,
    unwrap_state_dict,
)
from .vaaf import ViewAwareAttentionFusion
from .multiview import MultiImageHybridEVA, build_multiview_model
from .triple_branch import (
    FUSION_HEAD_TYPES,
    TripleBranchEVA,
    build_model_evax,
    build_triple_branch_model,
    load_triple_branch_model,
)

__all__ = [
    "EVA_X",
    "eva_x_base_patch16",
    "load_evax_init_weights",
    "load_single_view_weights",
    "unwrap_state_dict",
    "ViewAwareAttentionFusion",
    "MultiImageHybridEVA",
    "build_multiview_model",
    "FUSION_HEAD_TYPES",
    "TripleBranchEVA",
    "build_model_evax",
    "build_triple_branch_model",
    "load_triple_branch_model",
]
