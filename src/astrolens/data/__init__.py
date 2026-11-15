"""Physical data generation, FITS ingestion, photometric scaling, and exact D4 symmetries."""

from astrolens.data.dataset import FitsLensDataset, sanitize_byte_order
from astrolens.data.synthetic_fits import CLASS_NAMES, generate_synthetic_dataset
from astrolens.data.transforms import ArcsinhStretch, DihedralD4Transform

__all__ = [
    "ArcsinhStretch",
    "CLASS_NAMES",
    "DihedralD4Transform",
    "FitsLensDataset",
    "generate_synthetic_dataset",
    "sanitize_byte_order",
]