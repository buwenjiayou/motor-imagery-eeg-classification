from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy
import scipy
import sklearn
import tqdm
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_MAT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "mat"
    / "sub-001_ses-01_task_motorimagery_eeg.mat"
)


def main() -> None:
    print("python", sys.version.split()[0])
    print("numpy", numpy.__version__)
    print("scipy", scipy.__version__)
    print("sklearn", sklearn.__version__)
    print("matplotlib", matplotlib.__version__)
    print("tqdm", tqdm.__version__)

    mat = loadmat(SAMPLE_MAT)
    public_keys = sorted(key for key in mat.keys() if not key.startswith("__"))
    print("sample_mat", SAMPLE_MAT.name)
    print("keys", public_keys)
    print("data_shape", mat["data"].shape)
    print("labels_shape", mat["labels"].shape)


if __name__ == "__main__":
    main()
