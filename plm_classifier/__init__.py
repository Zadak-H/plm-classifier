"""PLM-Classifier: general sequence -> class label classification framework.

A config-driven, size-aware sibling of PLM-Regressor for classification:
- many protein language model embeddings + simple sequence encodings + extra tabular columns
- a large classifier zoo (classical + deep MLP/FNN/CNN), binary and multiclass
- Optuna search that adapts to dataset size (100 -> 1M+), stratified CV
- OOF model selection, probability outputs, ensemble/entropy uncertainty
- rich metrics (accuracy, F1, ROC-AUC, PR-AUC, MCC, balanced acc, log loss) + plots
- a Streamlit GUI and a thin CLI; ranks candidates by class probability
"""

__version__ = "0.1.0"
