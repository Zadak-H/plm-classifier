#!/usr/bin/env python3
"""Single source of truth for the classifier + pLM menus."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Tuple

from sklearn.base import BaseEstimator

from .models import classical as _classical
from .models import torch_models as _torch

# --------------------------------------------------------------------------- #
# Classifier registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                       # "classical" | "torch"
    available: bool
    max_n: Optional[int]
    requires_source: Optional[FrozenSet[str]] = None

    def eligible_for_n(self, n: int) -> bool:
        return self.available and (self.max_n is None or n <= self.max_n)


_POSITIONAL = frozenset({"onehot", "blosum62"})

_SPECS: List[ModelSpec] = [
    ModelSpec("logreg", "classical", True, None),
    ModelSpec("sgd", "classical", True, None),
    ModelSpec("lda", "classical", True, None),
    ModelSpec("gnb", "classical", True, None),
    ModelSpec("qda", "classical", True, 200_000),
    ModelSpec("knn", "classical", True, 50_000),
    ModelSpec("svc_rbf", "classical", True, 20_000),
    ModelSpec("gpc", "classical", True, 2_000),
    ModelSpec("rf", "classical", True, 200_000),
    ModelSpec("extra_trees", "classical", True, 200_000),
    ModelSpec("hist_gb", "classical", True, None),
    ModelSpec("xgboost", "classical", _classical.HAS_XGB, None),
    ModelSpec("lightgbm", "classical", _classical.HAS_LGB, None),
    ModelSpec("mlp", "classical", True, 50_000),
    ModelSpec("mlp_torch", "torch", _torch.HAS_TORCH, None),
    ModelSpec("cnn1d", "torch", _torch.HAS_TORCH, None, requires_source=_POSITIONAL),
]

MODEL_REGISTRY: Dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}

DEFAULT_MODELS: Tuple[str, ...] = (
    "logreg",
    "svc_rbf",
    "knn",
    "rf",
    "hist_gb",
    "mlp_torch",
)


def available_models() -> List[str]:
    return [name for name, spec in MODEL_REGISTRY.items() if spec.available]


def eligible_models(requested: List[str], n: int) -> List[str]:
    return [name for name in requested if (name in MODEL_REGISTRY and MODEL_REGISTRY[name].eligible_for_n(n))]


def build_model(
    name: str,
    trial: "object",
    random_state: int,
    use_gpu: bool,
    n_features: int,
    n_samples_train: int,
    n_classes: int,
) -> Tuple[str, BaseEstimator]:
    spec = MODEL_REGISTRY.get(name)
    if spec is None:
        raise ValueError(f"Unknown model '{name}'")
    if spec.kind == "torch":
        return _torch.build_torch_model(name, trial, random_state, use_gpu, n_features, n_samples_train, n_classes)
    return _classical.build_classical_model(name, trial, random_state, use_gpu, n_features, n_samples_train)


# --------------------------------------------------------------------------- #
# Protein language model registry (identical to PLM-Regressor)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PLMSpec:
    name: str
    backend: str
    model_id: str
    dim: int
    needs: str


_PLM_SPECS: List[PLMSpec] = [
    PLMSpec("esm2_8m", "esm", "esm2_t6_8M_UR50D", 320, "fair-esm"),
    PLMSpec("esm2_35m", "esm", "esm2_t12_35M_UR50D", 480, "fair-esm"),
    PLMSpec("esm2_150m", "esm", "esm2_t30_150M_UR50D", 640, "fair-esm"),
    PLMSpec("esm2", "esm", "esm2_t33_650M_UR50D", 1280, "fair-esm"),
    PLMSpec("esm2_3b", "esm", "esm2_t36_3B_UR50D", 2560, "fair-esm"),
    PLMSpec("esm2_15b", "esm", "esm2_t48_15B_UR50D", 5120, "fair-esm"),
    PLMSpec("esm1", "esm", "esm1_t34_670M_UR50S", 1280, "fair-esm"),
    PLMSpec("esm1b", "esm", "esm1b_t33_650M_UR50S", 1280, "fair-esm"),
    PLMSpec("esm1v", "esm", "esm1v_t33_650M_UR90S_1", 1280, "fair-esm"),
    PLMSpec("esmc_300m", "esmc", "esmc_300m", 960, "esm (evolutionaryscale)"),
    PLMSpec("esmc_600m", "esmc", "esmc_600m", 1152, "esm (evolutionaryscale)"),
    PLMSpec("esmplusplus_small", "hf_auto", "Synthyra/ESMplusplus_small", 960, "transformers"),
    PLMSpec("esmplusplus_large", "hf_auto", "Synthyra/ESMplusplus_large", 1152, "transformers"),
    PLMSpec("protT5", "t5", "Rostlab/prot_t5_xl_uniref50", 1024, "transformers+sentencepiece"),
    PLMSpec("protT5_half", "t5", "Rostlab/prot_t5_xl_half_uniref50-enc", 1024, "transformers+sentencepiece"),
    PLMSpec("protT5_bfd", "t5", "Rostlab/prot_t5_xl_bfd", 1024, "transformers+sentencepiece"),
    PLMSpec("protT5_xxl", "t5", "Rostlab/prot_t5_xxl_uniref50", 1024, "transformers+sentencepiece"),
    PLMSpec("prostT5", "t5", "Rostlab/ProstT5", 1024, "transformers+sentencepiece"),
    PLMSpec("protbert", "bert", "Rostlab/prot_bert", 1024, "transformers"),
    PLMSpec("protbert_bfd", "bert", "Rostlab/prot_bert_bfd", 1024, "transformers"),
    PLMSpec("ankh_base", "ankh", "ElnaggarLab/ankh-base", 768, "transformers"),
    PLMSpec("ankh_large", "ankh", "ElnaggarLab/ankh-large", 1536, "transformers"),
    PLMSpec("prosst", "prosst", "AI4Protein/ProSST-2048", 768, "transformers"),
    PLMSpec("carp_640m", "carp", "carp_640M", 1280, "sequence-models"),
]

PLM_REGISTRY: Dict[str, PLMSpec] = {spec.name: spec for spec in _PLM_SPECS}


def _backend_available(backend: str) -> bool:
    try:
        if backend == "esm":
            import esm  # noqa: F401

            return hasattr(__import__("esm"), "pretrained")
        if backend == "esmc":
            from esm.models.esmc import ESMC  # noqa: F401

            return True
        if backend == "carp":
            import sequence_models  # noqa: F401

            return True
        import transformers  # noqa: F401

        return True
    except Exception:
        return False


def available_plms() -> List[str]:
    return [name for name, spec in PLM_REGISTRY.items() if _backend_available(spec.backend)]
