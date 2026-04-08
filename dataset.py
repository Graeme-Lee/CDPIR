from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset


class MedicalSliceDataset(Dataset):
    """Dataset for 2D medical slices stored as .mat or .npy files.

    The dataset scans ``root`` recursively. If files are arranged in class
    subfolders, the first directory level under ``root`` is used as the class
    label. If files are stored directly under ``root``, all samples receive
    label 0.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        mat_key: str = "sub_label1",
        normalize: bool = True,
        extensions: Sequence[str] = (".mat", ".npy"),
    ) -> None:
        self.root = Path(root)
        self.mat_key = mat_key
        self.normalize = normalize
        self.extensions = tuple(ext.lower() for ext in extensions)

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.samples = self._scan_samples()
        if not self.samples:
            exts = ", ".join(self.extensions)
            raise RuntimeError(f"No samples with extensions [{exts}] found under {self.root}")

        self.class_to_idx = self._build_class_index(self.samples)
        self.indexed_samples: List[Tuple[Path, int]] = [
            (path, self.class_to_idx[class_name]) for path, class_name in self.samples
        ]

    def _scan_samples(self) -> List[Tuple[Path, str]]:
        files = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in self.extensions
        )

        samples: List[Tuple[Path, str]] = []
        for path in files:
            rel_parts = path.relative_to(self.root).parts
            class_name = rel_parts[0] if len(rel_parts) > 1 else "default"
            samples.append((path, class_name))
        return samples

    @staticmethod
    def _build_class_index(samples: Sequence[Tuple[Path, str]]) -> Dict[str, int]:
        class_names = sorted({class_name for _, class_name in samples})
        return {class_name: idx for idx, class_name in enumerate(class_names)}

    def _load_mat(self, path: Path) -> np.ndarray:
        data = loadmat(path)
        if self.mat_key not in data:
            available = ", ".join(sorted(k for k in data.keys() if not k.startswith("__")))
            raise KeyError(
                f"Key '{self.mat_key}' not found in {path}. Available keys: {available}"
            )
        return np.asarray(data[self.mat_key], dtype=np.float32)

    @staticmethod
    def _load_npy(path: Path) -> np.ndarray:
        return np.asarray(np.load(path), dtype=np.float32)

    def _load_array(self, path: Path) -> np.ndarray:
        if path.suffix.lower() == ".mat":
            array = self._load_mat(path)
        elif path.suffix.lower() == ".npy":
            array = self._load_npy(path)
        else:
            raise RuntimeError(f"Unsupported file extension: {path.suffix}")

        if array.ndim == 2:
            array = np.expand_dims(array, axis=0)
        elif array.ndim != 3:
            raise ValueError(
                f"Expected a 2D slice or a 3D tensor with channel dimension, got shape {array.shape}"
            )

        if self.normalize:
            array_min = float(array.min())
            array_max = float(array.max())
            array = (array - array_min) / (array_max - array_min + 1e-8)

        return np.ascontiguousarray(array, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.indexed_samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        path, label = self.indexed_samples[index]
        image = torch.from_numpy(self._load_array(path))
        return image, label
