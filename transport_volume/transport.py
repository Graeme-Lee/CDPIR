import torch as th
import torch
import numpy as np
import logging

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode
from leapctype import *
import matplotlib.pyplot as plt
import os
from pathlib import Path
from time import time
import gc
from scipy.io import loadmat
from tqdm import tqdm

device = 'cuda'
# leapct = tomographicModels()

class CTReconstructor:
    """
    Fan-beam CT Reconstructor using LeapCT geometry and operators.
    """

    def __init__(self, file_path, numAngles=55, numCols=821, numX=256, numY=256, numZ=3, pixelSize=1.09):
        self.gamma = 1
        self.eps = 1e-5
        self.numAngles = numAngles
        self.numCols = numCols
        self.numX = numX
        self.numY = numY
        self.numZ = numZ
        self.pixelSize = pixelSize

        # Load and normalize image
        self.img_stack = self._load_stack_and_normalize(file_path)
        # Setup geometry
        self.leapct = tomographicModels()
        self.leapct.print_warnings = False
        self.leapct.set_fanbeam(
            self.numAngles, self.numZ, self.numCols, self.pixelSize, self.pixelSize,
            0.5*(self.numZ-1), 0.5 * (self.numCols - 1),
            self.leapct.setAngleArray(self.numAngles, 360.0),
            625.61, 1097
        )
        # self.leapct.set_curvedDetector()
        # self.leapct.set_diameterFOV(999999)

        self.leapct.set_volume(self.numX, self.numY, self.numZ, voxelWidth=1.09, voxelHeight=1.09) # 
        # Projections and volume
        sino_np = self.leapct.allocateProjections()  # expect [A,Z,C]
        vol_np  = self.leapct.allocateVolume()
        # Torch tensors
        self.sino_img = torch.from_numpy(sino_np).to(device)   # [A,Z,C]
        self.ct_vol   = torch.from_numpy(vol_np).to(device)    # [Z,H,W]
        # Forward projection
        self.ct_gt_vol = torch.from_numpy(self.img_stack.astype(np.float32)).to(device)
        self.leapct.project(self.sino_img, self.ct_gt_vol)
        # Filters for reconstruction
        self.nfilters = filterSequence(1.0)
        self.nfilters.append(TV(self.leapct, delta=0.02/20.0))
        # Initial recon
        self.leapct.FBP(self.sino_img, self.ct_vol)
        self.leapct.ASDPOCS(self.sino_img, self.ct_vol, 30, 10, 0, self.nfilters)

    def _load_stack_and_normalize(self, file_paths):
        imgs = []
        for fp in file_paths:
            if fp.endswith(".mat"):
                data = loadmat(fp)
                img = data["sub_label1"]
            elif fp.endswith(".npy"):
                img = np.load(fp)
            else:
                raise ValueError(f"Unsupported file: {fp}")
            img = img.astype(np.float32)
            imgs.append(img)

        stack = np.stack(imgs, axis=0)  # [Z,H,W]

        # normalize (全stack一起 or 每slice单独都可以；先用全stack一起更稳定)
        stack = (stack - stack.min()) / (stack.max() - stack.min() + 1e-8)
        return np.ascontiguousarray(stack, dtype=np.float32)
    
    def cg_to_leap_vol(self, x):
        """
        x: [Z,1,H,W] or [Z,H,W]
        return vol: [Z,H,W] contiguous
        """
        if x.dim() == 4:
            vol = x[:, 0, ...]
        elif x.dim() == 3:
            vol = x
        else:
            raise ValueError(f"Unexpected x shape: {x.shape}")
        return vol.contiguous()
    
    def leap_vol_to_cg(self, vol):
        """
        vol: [Z,H,W]
        return x: [Z,1,H,W]
        """
        return vol.unsqueeze(1)

    def A(self, f):
        """Forward projector. x can be [Z,1,H,W] or [Z,H,W]. Return sino [A,Z,C]."""
        vol = self.cg_to_leap_vol(f)
        g = torch.from_numpy(self.leapct.allocateProjections()).to(vol.device)
        self.leapct.project(g, vol)
        return g

    def AT(self, g):
        """Adjoint-like backprojection. Return CG-shape [Z,1,H,W]."""
        vol = torch.from_numpy(self.leapct.allocateVolume()).to(g.device)
        self.leapct.FBP(g, vol)
        return vol

    def A_cg(self, x):
        """
        x: [Z,1,H,W]
        return: [Z,1,H,W]
        """
        return self.AT(self.A(x)).unsqueeze(1) + self.gamma * x

    # def CG(self, A_fn, b_cg, x, n_inner=10):
    #     """
    #     A_fn: [Z,1,H,W] -> [Z,1,H,W]
    #     b, x: [Z,1,H,W]
    #     """
    #     r = b_cg - A_fn(x)
    #     p = r
    #     rs_old = torch.matmul(r.reshape(1, -1), r.reshape(1, -1).T)


    #     for i in range(n_inner):
    #         Ap = A_fn(p)
    #         a = rs_old / torch.matmul(p.reshape(1, -1), Ap.reshape(1, -1).T)
    #         x += a * p
    #         r -= a * Ap
    #         rs_new = torch.matmul(r.reshape(1, -1), r.reshape(1, -1).T)
    #         if torch.sqrt(rs_new) < self.eps:
    #             break
    #         p = r + (rs_new / rs_old) * p
    #         rs_old = rs_new
    #     return x
    
    def CG(self, A_fn, b, x, n_inner=10):
        """
        A_fn: [Z,1,H,W] -> [Z,1,H,W]
        b, x: [Z,1,H,W]
        """
        r = b - A_fn(x)
        p = r.clone()
        rs_old = torch.sum(r * r)  # 标量

        for _ in range(n_inner):
            Ap = A_fn(p)
            denom = torch.sum(p * Ap) + 1e-12  # 标量
            a = rs_old / denom                 # 标量

            x = x + a * p
            r = r - a * Ap

            rs_new = torch.sum(r * r)          # 标量
            if torch.sqrt(rs_new).item() < self.eps:
                break

            p = r + (rs_new / (rs_old + 1e-12)) * p
            rs_old = rs_new

        return x

class sde:
    """SDE solver class"""
    def __init__(
        self, 
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
    ):
        assert t0 < t1, "SDE sampler has to be in forward time"

        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x
    
    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat # at last time point we do not perform the heun step

    def __forward_fn(self):
        """TODO: generalize here by adding all private functions ending with steps to it"""
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }

        try:
            sampler = sampler_dict[self.sampler_type]
        except:
            raise NotImplementedError("Smapler type not implemented.")
    
        return sampler

    def sample(self, init, model, **model_kwargs):
        """forward loop of sde"""

        # samples = []
        sampler = self.__forward_fn()
        input_dir = "/mnt/new_ssd/haodong/Datasets/Stanford_COCA/volume/slice"
        output_dir = "/home/haodong/Projects/SiT/COCA_VOLUME/rec_2.5D"
        ori_dir = "/home/haodong/Projects/SiT/COCA_VOLUME/ori"
        # fbp_dir = "/home/haodong/Projects/SiT/COCA-COCA/input"

        Z = 3
        half = Z // 2
        stride = Z - 2


        files = sorted(os.listdir(input_dir))

        
        for i_file in range(half, len(files) - half, stride):  # start from 1 because numslices=3 
            
            # fp0 = os.path.join(input_dir, files[i_file - 1])
            # fp1 = os.path.join(input_dir, files[i_file])
            # fp2 = os.path.join(input_dir, files[i_file + 1])

            window_names = [files[i_file + k] for k in range(-half, half + 1)]          # len=Z
            inner_names  = window_names#[1:-1]                                          # len=Z-2

            fps = [os.path.join(input_dir, n) for n in window_names]
            print(f"\n[Window Z={Z}] input: {', '.join(window_names)}")
            print(f"[Save inner]        : {', '.join(inner_names)}")

            ct_rec = CTReconstructor(fps, numAngles=55, numZ=Z)

            # --- init: 用 LEAP 初始重建作为 SiT/CG 初值 ---
            # ct_rec.ct_vol: [Z,H,W]  (LEAP space)
            # init / x / mean_x: [Z,1,H,W] (CG/SiT space)
            i = 0
            init = ct_rec.ct_vol.unsqueeze(1).contiguous()   # [3,1,256,256]
            x = init
            mean_x = init
            nfilters = filterSequence(1.0e0)
            nfilters.append(TV(ct_rec.leapct, delta=0.02/20.0))
            print(x.shape)
            start_time = time()

            center_base = os.path.splitext(files[i_file])[0]   # e.g. "003"
            file_without_extension = center_base               # 你后面 desc 和保存都用它
            t_list = self.t[:-1]
            pbar = tqdm(t_list, desc=f"Steps {file_without_extension}", unit="step", leave=False)           
            
            for step_idx, ti in enumerate(pbar):
                with th.no_grad():
                    x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)

                    # print(mean_x.shape)  #torch.Size([1, 1, 256, 256]) --> torch.Size([3, 1, 256, 256])

                    ATy = ct_rec.AT(ct_rec.sino_img)
                    b_cg = ATy.unsqueeze(1) + ct_rec.gamma * mean_x

                    # x_t = mean_x.squeeze()
                    # x_t = x_t[None,...]
                    # b_cg = ct_rec.AT(ct_rec.sino_img) + ct_rec.gamma *x_t   
                                      
                    x_cg =  ct_rec.CG(ct_rec.A_cg,  b_cg,mean_x, n_inner=10)
                    # print(x_cg.shape, mean_x.shape)  #torch.Size([1, 256, 256])  -->torch.Size([3, 1, 256, 256]) torch.Size([3, 1, 256, 256])

                    x_vol = x_cg[:, 0, ...].contiguous() 

                    # print(x_vol.shape) #torch.Size([1, 256, 256]) -->[3, 1, 256, 256]

                    ct_rec.leapct.ASDPOCS(ct_rec.sino_img,x_vol,10,5,0, nfilters)
                    mean_x = x_vol.unsqueeze(1)

                    x = mean_x

                    pbar.set_postfix({
                        "t": f"{ti:.3f}",
                        "mem": f"{torch.cuda.memory_allocated()/1024**2:.1f}MB"
                    })

                    # print(x.shape)  #torch.Size([1, 1, 256, 256])

                    # i = i+1
                    # print(i)
                    # samples.append(x)

            # normalize
            # mean_x = (mean_x - torch.min(mean_x)) / (torch.max(mean_x) - torch.min(mean_x) + 1e-8)

            # 1) 保存 center（可选）
            center_img = mean_x[half, 0].detach().cpu().numpy()
            np.save(f"{output_dir}/{center_base}_center.npy", np.abs(center_img))

            # 2) 保存 inner：每张单独存，并带上 center 标识（方案B）
            # inner_imgs = mean_x[1:Z-1, 0].detach().cpu().numpy()   # [Z-2,H,W]
            inner_imgs = mean_x.detach().cpu().numpy()   # [Z-2,H,W]
            for j, in_name in enumerate(inner_names):              # inner_names 是 002,003,004
                out_base = os.path.splitext(in_name)[0]            # "002"
                np.save(f"{output_dir}/{out_base}_from{center_base}.npy", np.abs(inner_imgs[j]))

            # 如果你仍然想保留一个堆叠版（可选）
            # np.save(f"{output_dir}/{center_base}_innerStack.npy", np.abs(inner_imgs))
        
            # plt.imsave(f"{output_dir}/{file_without_extension}.png", np.abs(mean_x.squeeze().cpu().numpy()), cmap='gray')
            print(f"{file_without_extension} has been sampled. Sampling took {time() - start_time:.2f} seconds.")

            del mean_x, x, init, x_cg, x_vol, b_cg, ATy, ct_rec
            gc.collect()
            torch.cuda.empty_cache()

        return mean_x
    

class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)

class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()

class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(
        self,
        *,
        model_type,
        path_type,
        loss_type,
        train_eps,
        sample_eps,
    ):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        '''
            Standard multivariate normal prior
            Assume z is batched
        '''
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)
    

    def check_interval(
        self, 
        train_eps, 
        sample_eps, 
        *, 
        diffusion_form="SBDM",
        sde=False, 
        reverse=False, 
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if (type(self.path_sampler) in [path.VPCPlan]):

            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) \
            and (self.model_type != ModelType.VELOCITY or sde): # avoid numerical issue by taking a first semi-implicit step

            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        
        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1


    def sample(self, x1):
        """Sampling x0 & t based on shape of x1 (if needed)
          Args:
            x1 - data point; [batch, *dim]
        """
        
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1
    

    def training_losses(
        self, 
        model,  
        x1, 
        model_kwargs=None
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """
        if model_kwargs == None:
            model_kwargs = {}
        
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)

        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
        else: 
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
                
        return terms
    

    def get_drift(
        self
    ):
        """member function for obtaining the drift of the probability flow ODE"""
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return (-drift_mean + drift_var * model_output) # by change of variable
        
        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return (-drift_mean + drift_var * score)
        
        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode
        
        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn
    

    def get_score(
        self,
    ):
        """member function for obtaining score of 
            x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
        else:
            raise NotImplementedError()
        
        return score_fn


class Sampler:
    """Sampler class for the transport model"""
    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """
        
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()
    
    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):

        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion
        
        sde_drift = \
            lambda x, t, model, **kwargs: \
                self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(x, t, model, **kwargs)
    
        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion
    
    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""
    
        if last_step is None:
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x
        elif last_step == "Mean":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + sde_drift(x, t, model, **model_kwargs) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = \
                lambda x, t, model, **model_kwargs: \
                    x + self.drift(x, t, model, **model_kwargs) * last_step_size
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.0,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)
            

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs, ts, model, **model_kwargs)
            xs.append(x)

            #assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        - reverse: whether solving the ODE in reverse (data to noise); default to False
        """
        if reverse:
            drift = lambda x, t, model, **kwargs: self.drift(x, th.ones_like(t) * (1 - t), model, **kwargs)
        else:
            drift = self.drift

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )
        
        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps: 
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)
        
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
