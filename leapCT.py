
import torch
import numpy as np

from leapctype import *
import matplotlib.pyplot as plt
import os
from pathlib import Path
from time import time
import gc
from scipy.io import loadmat
from PIL import Image
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# leapct = tomographicModels()


class CTReconstructor:
    """
    Fan-beam CT Reconstructor using LeapCT geometry and operators.
    """

    def __init__(self, file_path, numAngles=55, numCols=821, numX=256, numY=256, pixelSize=1.09):
        self.gamma = 1
        self.eps = 1e-5
        self.numAngles = numAngles
        self.numCols = numCols
        self.numX = numX
        self.numY = numY
        self.pixelSize = pixelSize

        # Load and normalize image
        self.img = self._load_and_normalize(file_path)
        # Setup geometry
        self.leapct = tomographicModels()
        self.leapct.set_fanbeam(
            self.numAngles, 1, self.numCols, self.pixelSize, self.pixelSize,
            0.0, 0.5 * (self.numCols - 1),
            self.leapct.setAngleArray(self.numAngles, 360.0),
            625.61, 1097
        )
        # self.leapct.set_curvedDetector()
        # self.leapct.set_diameterFOV(999999)

        self.leapct.set_volume(self.numX, self.numY, 1, voxelWidth=1.09, voxelHeight=1.09) # 
        # Projections and volume
        self.sino = self.leapct.allocateProjections()
        self.ct = self.leapct.allocateVolume()
        # Torch tensors
        self.ct_img = torch.from_numpy(self.img[None, :, :].astype(np.float32)).to(device)
        self.sino_img = torch.from_numpy(self.sino).to(device)
        # Forward projection
        self.leapct.project(self.sino_img, self.ct_img)
        self.at_sino = None
        # Filters for reconstruction
        self.nfilters = filterSequence(1.0)
        self.nfilters.append(TV(self.leapct, delta=0.02/20.0))
        # Initial recon
        self.leapct.FBP(self.sino_img, self.ct_img)
        self.leapct.ASDPOCS(self.sino_img, self.ct_img, 30, 10, 0, self.nfilters)
        self.at_sino = self.AT(self.sino_img)

    def apply_data_consistency(
        self,
        x,
        use_cg=1,
        use_asd_pocs=1,
        cg_inner=10,
        asd_pocs_iters=10,
        asd_pocs_subsets=5,
    ):
        """Apply data consistency to the current reconstruction estimate."""
        x_dc = x

        if int(use_cg) == 1:
            b_cg = self.at_sino + self.gamma * x_dc
            x_dc = self.CG(self.A_cg, b_cg, x_dc, n_inner=cg_inner)

        if int(use_asd_pocs) == 1:
            self.leapct.ASDPOCS(self.sino_img, x_dc, asd_pocs_iters, asd_pocs_subsets, 0, self.nfilters)

        return x_dc

    def _load_and_normalize(self, file_path):
        suffix = Path(file_path).suffix.lower()
        if suffix == '.mat':
            data = loadmat(file_path)
            img = data['sub_label1']
        elif suffix == '.npy':
            img = np.load(file_path)
        elif suffix in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
            img = np.array(Image.open(file_path).convert('F'))
        else:
            raise ValueError(f"Unsupported input file type: {suffix}")
        img = img.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        return np.ascontiguousarray(img, dtype=np.float32)

    def A(self, f):
        """Forward projector (A)."""
        g = self.leapct.allocateProjections()
        g = torch.from_numpy(g).to(device)
        self.leapct.project(g, f)
        return g

    def AT(self, g):
        """Backprojection operator (A^T), via FBP."""
        f = self.leapct.allocateVolume()
        f = torch.from_numpy(f).to(device)
        self.leapct.FBP(g, f)
        return f

    def A_cg(self, x):
        """CG system operator: (A^T A + gamma*I)."""
        return self.AT(self.A(x)) + self.gamma * x

    def CG(self, A_fn, b_cg, x, n_inner=10):
        """Conjugate Gradient solver."""
        r = b_cg - A_fn(x)
        p = r
        rs_old = torch.matmul(r.reshape(1, -1), r.reshape(1, -1).T)
        for i in range(n_inner):
            Ap = A_fn(p)
            a = rs_old / torch.matmul(p.reshape(1, -1), Ap.reshape(1, -1).T)
            x += a * p
            r -= a * Ap
            rs_new = torch.matmul(r.reshape(1, -1), r.reshape(1, -1).T)
            if torch.sqrt(rs_new) < self.eps:
                break
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        return x