import sys
# sys.path.append('/home/songjian/project/MambaCD')
# sys.path.append("/storage/alperengenc/change_detection/ChangeMamba_AG/")
sys.path.append("/mnt/storage1/alpgenc/change_detection/ChangeMamba_AG/")

from datetime import datetime

import argparse
import os
import time

import numpy as np

from changedetection.configs.config import get_config

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from changedetection.datasets.make_data_loader import DamageAssessmentDatset, make_data_loader
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.ChangeMambaBDA import ChangeMambaBDA
import imageio
import numpy as np
import seaborn as sns

import logging
import json
import copy
import matplotlib.pyplot as plt
import matplotlib.image
from dataclasses import dataclass

from changedetection.models.alignment_module import AlignmentHead, AlignmentArgs, warp_image
from changedetection.models.attn_gate import AttentionGateArgs, AttentionGate2d


ori_label_value_dict = {
    'background': (0, 0, 0),
    'no_damage': (70, 181, 121),
    'minor_damage': (167, 187, 27),
    'major_damage': (228, 189, 139),
    'destroy': (181, 70, 70)
}

target_label_value_dict = {
    'background': 0,
    'no_damage': 1,
    'minor_damage': 2,
    'major_damage': 3,
    'destroy': 4,
}

def map_labels_to_colors(labels, ori_label_value_dict, target_label_value_dict):
    # Reverse the target_label_value_dict to get a mapping from target labels to original labels
    target_to_ori = {v: k for k, v in target_label_value_dict.items()}
    
    # Initialize an empty 3D array for the color-mapped labels
    H, W = labels.shape
    color_mapped_labels = np.zeros((H, W, 3), dtype=np.uint8)
    
    for target_label, ori_label in target_to_ori.items():
        # Find where the label matches the current target label
        mask = labels == target_label
        
        # Map these locations to the corresponding color value
        color_mapped_labels[mask] = ori_label_value_dict[ori_label]
    
    return color_mapped_labels


def register_attn_hooks(model):
    attn_maps = {}

    def make_hook(name):
        def hook_fn(module, input, output):
            if hasattr(module, "last_alpha"):
                attn_maps[name] = module.last_alpha.cpu()
        return hook_fn

    handles: list[torch.utils.hooks.RemovableHandle] = []
    for name, m in model.named_modules():
        if isinstance(m, AttentionGate2d):
            h = m.register_forward_hook(make_hook(name))
            handles.append(h)
            logging.info(f"[Hook registered on] {name} ({id(m)=})")
    return attn_maps, handles

@dataclass
class AlignmentCapture:
    """Stores tensors for ONE forward pass (per module)."""
    flow: torch.Tensor          # [1, 2, H_f, W_f] on CPU
    f_pre: torch.Tensor         # [1, C, H_f, W_f] on CPU
    f_post: torch.Tensor        # [1, C, H_f, W_f] on CPU
    f_pre_warp: torch.Tensor    # [1, C, H_f, W_f] on CPU

def register_alignment_hooks(model: torch.nn.Module) -> tuple[dict[str, AlignmentCapture], list[torch.utils.hooks.RemovableHandle]] :
    captures: dict[str, AlignmentCapture] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(module_name: str):
        def hook(module: torch.nn.Module, input: tuple[torch.Tensor, torch.Tensor], output: tuple[torch.Tensor, torch.Tensor] ):
            """
            module: instance of `AlignmentHead` <br/>
            input: AlignmentHead inputs, `f_pre` and `f_post` <br/>
            output: AlignmentHead outputs, `f_pre_warp` and `flow`
            """
            f_pre, f_post = input  # both are [B, C, Hf, Wf]
            f_pre_warp, flow = output  # flow is [B, 2, Hf, Wf]

            # save the first element in the batch
            captures[module_name] = AlignmentCapture(
                flow = flow[:1].detach().float().cpu(),
                f_pre = f_pre[:1].detach().float().cpu(),
                f_post = f_post[:1].detach().float().cpu(),
                f_pre_warp = f_pre_warp[:1].detach().float().cpu(),
            )

        return hook

    for name, m in model.named_modules():
        if isinstance(m, AlignmentHead):
            h = m.register_forward_hook(make_hook(name))
            handles.append(h)
            logging.info(f"[Hook registered on] {name} ({id(m)=})")

    return captures, handles


def denormalize_img(t: torch.Tensor, mean: list[float], std: list[float]) -> np.ndarray:
    """
    t: [3,H,W] torch tensor in normalized range
    mean, std: per-channel lists
    returns: [H,W,3] uint8 image 0–255
    """
    t = t.clone().cpu()
    for c in range(3):
        t[c] = t[c] * std[c] + mean[c]
    arr = t.permute(1,2,0).numpy()
    arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
    return arr


def save_all_attn_maps(pre_img: np.ndarray, post_img: np.ndarray, pre_mask, post_mask, attn_maps: dict[str, torch.Tensor], out_path: str):
    """
    img: [H,W,3] uint8
    mask: [H,W] numpy int
    attn_maps: dict { "building.ag1": tensor[B,1,h,w], ... }
    out_path: str
    """
    building_maps = {}
    damage_maps = {}
    
    for name, tensor in attn_maps.items():
        if 'decoder_building' in name: building_maps[name] = tensor
        elif 'decoder_damage' in name: damage_maps[name] = tensor
    
    # Sort by gate number (ag3 -> ag2 -> ag1)
    def sort_key(item):
        name = item[0]
        if 'ag3' in name: return 0
        elif 'ag2' in name: return 1
        elif 'ag1' in name: return 2
        return 999
    building_maps = dict(sorted(building_maps.items(), key=sort_key))
    damage_maps = dict(sorted(damage_maps.items(), key=sort_key))

    if pre_mask is not None:
        H, W = pre_mask.shape
    elif post_mask is not None:
        H, W = post_mask.shape

    fig, axs = plt.subplots(2, 5, figsize=(20, 8))

    def get_heatmap(attn_tensor: torch.Tensor) -> np.ndarray:
        heat = F.interpolate(attn_tensor, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat
    
    def display_heatmaps(attn_maps: dict, row_idx: int, img: np.ndarray, mask: torch.Tensor, mask_title: str):
        # Column 0: Image
        axs[row_idx, 0].imshow(img)
        axs[row_idx, 0].set_title("Input Image", fontsize=10)
        axs[row_idx, 0].axis("off")
        # Column 1: GT mask
        axs[row_idx, 1].imshow(mask, cmap="gray")
        axs[row_idx, 1].set_title(mask_title, fontsize=10)
        axs[row_idx, 1].axis("off")
        for col_idx, (name, attn_tensor) in enumerate(attn_maps.items(), start=2):
            heat = get_heatmap(attn_tensor)
            axs[row_idx, col_idx].imshow(mask, cmap="gray")
            axs[row_idx, col_idx].imshow(heat, cmap="jet", alpha=0.6)
            axs[row_idx, col_idx].set_title(f"{name} on {mask_title}")
            axs[row_idx, col_idx].axis("off")

    if pre_mask is not None:
        display_heatmaps(building_maps, row_idx=0, img=pre_img, mask=pre_mask, mask_title="Building GT")

    if post_mask is not None:
        display_heatmaps(damage_maps, row_idx=1, img=post_img, mask=post_mask, mask_title="Damage GT")

    plt.tight_layout()
    plt.savefig(f"{out_path}", dpi=150)
    plt.close(fig)
    logging.info(f"Saved attention visualization: {out_path}")


def save_alignment_visualizations(pre_img: np.ndarray, post_img: np.ndarray, capture: AlignmentCapture, out_path: str, arrows_stride: int = 1):
    dx = capture.flow[0, 0].detach().float().cpu().numpy()
    dy = capture.flow[0, 1].detach().float().cpu().numpy()
    flow_magnitudes = np.sqrt(dx ** 2 + dy ** 2)

    Hf, Wf = flow_magnitudes.shape
    yy, xx = np.mgrid[0:Hf:arrows_stride, 0:Wf:arrows_stride]

    similarity_without_alignment = F.cosine_similarity(capture.f_pre, capture.f_post, dim=1)[0].cpu().numpy()
    similarity_with_alignment = F.cosine_similarity(capture.f_pre_warp, capture.f_post, dim=1)[0].cpu().numpy()

    ## metric 1
    average_gain = similarity_with_alignment.mean() - similarity_without_alignment.mean()
    logging.info(f"   {average_gain = }")

    ## metric 2
    mse_without_alignment = F.mse_loss(capture.f_pre, capture.f_post)
    mse_with_alignment = F.mse_loss(capture.f_pre_warp, capture.f_post)
    logging.info(f"   Error reduced from {mse_without_alignment=} to {mse_with_alignment=} (error diff: {mse_without_alignment - mse_with_alignment})")
    
    ## metric 3
    # f_pre_reduced        = torch.sqrt((capture.f_pre * capture.f_pre).sum(dim=1))[0]
    # f_pre_warped_reduced = torch.sqrt((capture.f_pre_warp * capture.f_pre_warp).sum(dim=1))[0]
    # f_post_reduced       = torch.sqrt((capture.f_post * capture.f_post).sum(dim=1))[0]
    
    # diff_before   = (f_pre_reduced - f_post_reduced).abs().detach().float().cpu().numpy()
    # diff_after    = (f_pre_warped_reduced - f_post_reduced).abs().detach().float().cpu().numpy()
    # delta_improve = diff_before - diff_after

    diff_before_t = (capture.f_pre - capture.f_post).abs().mean(dim=1)      # [1, Hf, Wf]
    diff_after_t  = (capture.f_pre_warp - capture.f_post).abs().mean(dim=1) # [1, Hf, Wf]

    diff_before   = diff_before_t[0].detach().float().cpu().numpy()
    diff_after    = diff_after_t[0].detach().float().cpu().numpy()
    delta_improve = diff_before - diff_after


    def robust_vmax(a: np.ndarray, b: np.ndarray, p: float = 99.0) -> float:
        """Shared vmax for two positive maps, to make before/after comparable."""
        x = np.concatenate([a.reshape(-1), b.reshape(-1)])
        vmax = float(np.percentile(x, p))
        return max(vmax, 1e-6)
    def robust_sym_v(delta: np.ndarray, p: float = 99.0) -> float:
        """Symmetric range for signed delta maps."""
        lo, hi = np.percentile(delta.reshape(-1), [100 - p, p])
        v = float(max(abs(lo), abs(hi)))
        return max(v, 1e-6)

    vmax_diff = robust_vmax(diff_before, diff_after, p=99.0)
    v_delta = robust_sym_v(delta_improve, p=99.0)

    # --- Plotting
    fig, axs = plt.subplots(3, 3, figsize=(14, 13))
    axs = axs.flatten()

    axs[0].imshow(pre_img)
    axs[0].set_title("Pre-disaster image")
    axs[0].axis("off")

    axs[1].imshow(post_img)
    axs[1].set_title("Post-disaster image")
    axs[1].axis("off")

    im2 = axs[2].imshow(flow_magnitudes)
    axs[2].set_title("Flow magnitude (feature scale)")
    axs[2].axis("off")
    fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)

    # Flow arrows on magnitude
    axs[3].imshow(flow_magnitudes)
    axs[3].quiver(xx, yy, dx[yy, xx], dy[yy, xx], angles="xy", scale_units="xy", scale=1.0)
    axs[3].set_title("Flow field (dx, dy)")
    axs[3].axis("off")

    im4 = axs[4].imshow(similarity_without_alignment, vmin=-1, vmax=1)
    axs[4].set_title("Cosine similarity without alignment")
    axs[4].axis("off")
    fig.colorbar(im4, ax=axs[4], fraction=0.046, pad=0.04)

    im5 = axs[5].imshow(similarity_with_alignment, vmin=-1, vmax=1)
    axs[5].set_title("Cosine similarity with alignment")
    axs[5].axis("off")
    fig.colorbar(im5, ax=axs[5], fraction=0.046, pad=0.04)

    im6 = axs[6].imshow(diff_before, vmin=0, vmax=vmax_diff, cmap="magma")
    axs[6].set_title("$F^{pre} - F^{post}$")
    axs[6].axis("off")
    fig.colorbar(im6, ax=axs[6], fraction=0.046, pad=0.04)

    im7 = axs[7].imshow(diff_after, vmin=0, vmax=vmax_diff, cmap="magma")
    axs[7].set_title("$F_{warped}^{pre} - F^{post}$")
    axs[7].axis("off")
    fig.colorbar(im7, ax=axs[7], fraction=0.046, pad=0.04)

    im8 = axs[8].imshow(delta_improve, vmin=-v_delta, vmax=v_delta, cmap="coolwarm")
    axs[8].set_title("ΔDiff = Before - After")
    axs[8].axis("off")
    fig.colorbar(im8, ax=axs[8], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)



class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)
        self.total_evaluator_loc = Evaluator(num_class=2)
        self.total_evaluator_clf = Evaluator(num_class=5)

        if args.enable_alignment:
            alignment_args = AlignmentArgs(enabled=True, stages=(1,2,), mid_ch=64)
        else:
            alignment_args = AlignmentArgs(enabled=False, stages=None, mid_ch=None)
        logging.info(f" > ALIGNMENT params: {alignment_args = }")

        attn_gate_args = AttentionGateArgs(enable_building_ag = args.enable_attn_gate_building, enable_damage_ag=args.enable_attn_gate_damage)
        logging.info(f" > ATTENTION GATE params: {attn_gate_args = }")

        self.deep_model = ChangeMambaBDA(
            output_building=2, output_damage=5,
            pretrained=args.pretrained_weight_path,
            alignment_args=alignment_args,
            attn_gate_args=attn_gate_args,
            patch_size=config.MODEL.VSSM.PATCH_SIZE, 
            in_chans=config.MODEL.VSSM.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.VSSM.DEPTHS, 
            dims=config.MODEL.VSSM.EMBED_DIM, 
            # ===================
            ssm_d_state=config.MODEL.VSSM.SSM_D_STATE,
            ssm_ratio=config.MODEL.VSSM.SSM_RATIO,
            ssm_rank_ratio=config.MODEL.VSSM.SSM_RANK_RATIO,
            ssm_dt_rank=("auto" if config.MODEL.VSSM.SSM_DT_RANK == "auto" else int(config.MODEL.VSSM.SSM_DT_RANK)),
            ssm_act_layer=config.MODEL.VSSM.SSM_ACT_LAYER,
            ssm_conv=config.MODEL.VSSM.SSM_CONV,
            ssm_conv_bias=config.MODEL.VSSM.SSM_CONV_BIAS,
            ssm_drop_rate=config.MODEL.VSSM.SSM_DROP_RATE,
            ssm_init=config.MODEL.VSSM.SSM_INIT,
            forward_type=config.MODEL.VSSM.SSM_FORWARDTYPE,
            # ===================
            mlp_ratio=config.MODEL.VSSM.MLP_RATIO,
            mlp_act_layer=config.MODEL.VSSM.MLP_ACT_LAYER,
            mlp_drop_rate=config.MODEL.VSSM.MLP_DROP_RATE,
            # ===================
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            patch_norm=config.MODEL.VSSM.PATCH_NORM,
            norm_layer=config.MODEL.VSSM.NORM_LAYER,
            downsample_version=config.MODEL.VSSM.DOWNSAMPLE,
            patchembed_version=config.MODEL.VSSM.PATCHEMBED,
            gmlp=config.MODEL.VSSM.GMLP,
            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
        ) 
        self.deep_model = self.deep_model.cuda()
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        self.building_map_T1_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'building_localization_map')
        self.change_map_T2_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'damage_classification_map')
        self.attention_map_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'attention_map')
        self.alignment_visualization_saved_path = os.path.join(args.result_saved_path, args.dataset, args.model_type, 'alignment')

        if self.args.save_output_images:
            if not os.path.exists(self.building_map_T1_saved_path):
                os.makedirs(self.building_map_T1_saved_path)
            if not os.path.exists(self.change_map_T2_saved_path):
                os.makedirs(self.change_map_T2_saved_path)

        if self.args.save_attention_images:
            if not os.path.exists(self.attention_map_saved_path):
                os.makedirs(self.attention_map_saved_path)
        if self.args.save_alignment_images:
            if not os.path.exists(self.alignment_visualization_saved_path):
                os.makedirs(self.alignment_visualization_saved_path)


        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model_dict = {}
            state_dict = self.deep_model.state_dict()
            for k, v in checkpoint.items():
                if k in state_dict:
                    model_dict[k] = v
            state_dict.update(model_dict)
            self.deep_model.load_state_dict(state_dict)

        self.deep_model.eval()

        if self.args.measure_efficiency:
            #* use measure_model_efficiency2 for more consistent results with the ChangeMamba paper
            # logging.info("measure_model_efficiency_1")
            # self.measure_model_efficiency()
            logging.info("measure_model_efficiency_2")
            self.measure_model_efficiency2()

    def measure_model_efficiency(self):
        """
        DEPRECATED: Use measure_model_efficiency2 instead, since it is more consistent with the ChangeMamba paper.
        """
        logging.info("=" * 80)
        logging.info("MODEL EFFICIENCY MEASUREMENTS")
        logging.info("=" * 80)

        self.log_model_param_count()

        logging.info(" --- " * 10)

        imsize = 512
        batchsize = 1
        try:
            logging.info(f"FLOPs: Method 1 (imp)")
            flops_gflops = self.measure_flops(imsize=imsize)
            logging.info(f"FLOPs: {flops_gflops:.2f} GFLOPs (input: {imsize} x {imsize})")
        except Exception as e:
            logging.warning(f"FLOPs Method 1 measurement failed: {e}")
        logging.info(" ***** " * 3)
        try:
            logging.info(f"FLOPs: Method 2 (analyze.get_flops)")
            from analyze.get_flops import fvcore_flop_count, supported_ops as fvcore_supported_ops
            # params, flops = fvcore_flop_count(self.deep_model, input_shape=(3, imsize, imsize))
            flops_gflops = self.measure_flops(imsize=imsize, supported_ops=fvcore_supported_ops)
            logging.info(f"FLOPs: {flops_gflops:.2f} GFLOPs (input: {imsize} x {imsize})")
        except Exception as e:
            logging.warning(f"FLOPs Method 2 measurement failed: {e}")

        logging.info(" --- " * 10)

        try:
            throughput_dict = self.measure_throughput(imsize=imsize, batchsize=batchsize)
            logging.info(f"Throughput: {throughput_dict['imgs_per_sec']:.2f} images/sec")
            logging.info(f"Latency: {throughput_dict['ms_per_image']:.2f} ms/image")
            logging.info(f"  (batch_size={throughput_dict['batch_size']}, measured over {throughput_dict['iterations']} iterations)")
        except Exception as e:
            logging.warning(f"Throughput measurement failed: {e}")
        
        logging.info("=" * 80)

    def log_model_param_count(self):
        total_params = sum(p.numel() for p in self.deep_model.parameters())
        logging.info(f"Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        
    def measure_flops(self, imsize: int, supported_ops: dict = None) -> float:
        """Measure FLOPs for single image pair"""
        from classification.models.vmamba import selective_scan_flop_jit 
        from fvcore.nn.flop_count import flop_count

        # Single image for FLOPs calculation
        H = W = imsize
        pre_img = torch.randn(1, 3, H, W).cuda()
        post_img = torch.randn(1, 3, H, W).cuda()
        
        if supported_ops is None:
            supported_ops = {
                "aten::silu": None,
                "aten::neg": None,
                "aten::exp": None,
                "aten::flip": None,
                "prim::PythonOp.SelectiveScanFn": selective_scan_flop_jit,
            }
        with torch.no_grad():
            Gflops, unsupported = flop_count(
                model=self.deep_model,
                inputs=(pre_img, post_img),
                supported_ops=supported_ops
            )
        total_gflops = sum(Gflops.values())
        if unsupported:
            logging.warning(f"Unsupported ops in FLOPs calculation ({len(unsupported)}): {list(unsupported.keys())}")
        return total_gflops

    def measure_throughput(self, imsize: int, batchsize: int, num_iterations: int = 100, warmup: int = 10) -> dict:
        """Measure inference throughput"""
        H = W = imsize
        batch_size = batchsize

        # Create dummy batch
        pre_img = torch.randn(batch_size, 3, H, W).cuda()
        post_img = torch.randn(batch_size, 3, H, W).cuda()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = self.deep_model(pre_img, post_img)
        
        torch.cuda.synchronize()
        
        # Measure
        start_time = time.time()
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.deep_model(pre_img, post_img)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        
        total_images = num_iterations * batch_size
        imgs_per_sec = total_images / elapsed
        ms_per_batch = (elapsed / num_iterations) * 1000
        ms_per_image = ms_per_batch / batch_size
        
        return {
            'imgs_per_sec': imgs_per_sec,
            'ms_per_batch': ms_per_batch,
            'ms_per_image': ms_per_image,
            'batch_size': batch_size,
            'iterations': num_iterations
        }


    def measure_model_efficiency2(self):
        from thop import profile as thop_profile, clever_format as thop_clever_format
        from analyze.get_flops import supported_ops as fvcore_supported_ops

        logging.info("=" * 80)
        logging.info("MODEL EFFICIENCY MEASUREMENTS")
        logging.info("=" * 80)

        imsize = 512
        batchsize = 1
        H = W = imsize
        pre_img = torch.randn(1, 3, H, W).cuda()
        post_img = torch.randn(1, 3, H, W).cuda()
        macs, params = thop_profile(model=self.deep_model, inputs=(pre_img, post_img), custom_ops=fvcore_supported_ops, verbose=True)
        macs, params = thop_clever_format([macs, params], "%.2f")
        logging.info(f"THOP FLOPs: {macs}  Parameters: {params} (input size: {W} x {H})")

        try:
            throughput_dict = self.measure_throughput(imsize=imsize, batchsize=batchsize)
            logging.info(f"Throughput: {throughput_dict['imgs_per_sec']:.2f} images/sec")
            logging.info(f"Latency: {throughput_dict['ms_per_image']:.2f} ms/image")
            logging.info(f"  (batch_size={throughput_dict['batch_size']}, measured over {throughput_dict['iterations']} iterations)")
        except Exception as e:
            logging.warning(f"Throughput measurement failed: {e}")

        logging.info("=" * 80)

    def infer(self):
        torch.cuda.empty_cache()
        if self.args.extension is None:
            ext = "tif" if 'mwBTFreddy' in self.args.dataset else "png"
        else: 
            ext = self.args.extension
        dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test', extension=ext)
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()
        self.total_evaluator_loc.reset()
        self.total_evaluator_clf.reset()          
        # vbar = tqdm(val_data_loader, ncols=50)

        if self.args.save_attention_images:
            attn_maps, attn_hook_handles = register_attn_hooks(self.deep_model)

        if self.args.save_alignment_images:
            alignment_captures, alignment_hook_handles = register_alignment_hooks(self.deep_model)

        with torch.no_grad():
            for itera, data in enumerate(tqdm(val_data_loader)):
                if itera % 10 == 0:

                    loc_f1_score = self.total_evaluator_loc.Pixel_F1_score()
                    damage_f1_score = self.total_evaluator_clf.Damage_F1_socore()
                    harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
                    oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

                    log = f'inference: {itera:>4}/{len(val_data_loader):>4} | Current F1_overall: {oaf1 * 100:.3f}% (Clsf: {harmonic_mean_f1 * 100:.3f}%, Loc: {loc_f1_score * 100:.3f}%) [cumulative]'
                    logging.info(log)

                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, names = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()

                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)



                # --- visualize AG map for this sample ---
                if itera % 10 == 0 and self.args.save_attention_images and (self.args.enable_attn_gate_building or self.args.enable_attn_gate_damage):
                    building_available = labels_loc.max().item() != 0
                    if not building_available:
                        logging.info(f" > No building in {names[0]}, skipping attention visualization.")
                        pass
                    elif len(attn_maps) > 0:
                        pre_img = None
                        post_img = None
                        pre_mask = None
                        post_mask = None
                        # save_all_attn_maps(img, mask, attn_maps, os.path.join(self.attention_map_saved_path, f"{names[0]}_all.png"))
                        if self.args.enable_attn_gate_building:
                            pre_mask = labels_loc[0].detach().cpu().numpy()
                            pre_img = denormalize_img(pre_change_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        if self.args.enable_attn_gate_damage:
                            post_mask = labels_clf[0].detach().cpu().numpy()
                            post_mask[post_mask == 255] = 0
                            post_img = denormalize_img(post_change_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        save_all_attn_maps(pre_img, post_img, pre_mask, post_mask, attn_maps, os.path.join(self.attention_map_saved_path, f"{names[0]}_attentions.png"))
                        # if itera > 10:
                        #     break  # DEBUG (quick results, only visualize first n samples)


                # --- visualize Alignment for this sample ---
                if self.args.save_alignment_images and self.args.enable_alignment: # and itera % 10 == 0:
                    building_available = labels_loc.max().item() != 0
                    if not building_available:
                        logging.info(f" > No building in {names[0]}, skipping ALIGNMENT visualization.")
                        pass

                    elif len(alignment_captures) > 0:
                        for i, module_name in enumerate(alignment_captures.keys(), start=1):
                            alignment_visualization_output_img_name = f"{names[0]}_{module_name}_({i})"
                            logging.info(f" > Alignment visualization: {alignment_visualization_output_img_name}")
                            cap: AlignmentCapture = alignment_captures[module_name]
                            pre_img = denormalize_img(pre_change_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            post_img = denormalize_img(post_change_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            save_alignment_visualizations(
                                pre_img, post_img, cap, 
                                os.path.join(self.alignment_visualization_saved_path, f"{alignment_visualization_output_img_name}.png"),
                                arrows_stride = int(2/i)
                            )

                            # warped_pre_image = warp_image(pre_change_imgs, cap.flow)
                            # warped_pre_image_denorm = denormalize_img(warped_pre_image[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                            # matplotlib.image.imsave(
                            #     os.path.join(self.alignment_visualization_saved_path, f"{names[0]}_original_{module_name}_({i}).png"), 
                            #     pre_img
                            # )
                            # matplotlib.image.imsave(
                            #     os.path.join(self.alignment_visualization_saved_path, f"{names[0]}_warped_{module_name}_({i}).png"), 
                            #     warped_pre_image_denorm
                            # )


                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.total_evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf_eval = output_clf[labels_loc > 0]
                labels_clf_eval = labels_clf[labels_loc > 0]
                self.total_evaluator_clf.add_batch(labels_clf_eval, output_clf_eval)

                if itera % 10 == 0 and self.args.save_output_images:
                    image_name = names[0] + '.png'

                    output_loc = np.squeeze(output_loc)
                    output_loc[output_loc > 0] = 255

                    output_clf = map_labels_to_colors(np.squeeze(output_clf), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
                    output_clf[output_loc == 0] = 0

                    imageio.imwrite(os.path.join(self.building_map_T1_saved_path, image_name), output_loc.astype(np.uint8))
                    imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image_name), output_clf.astype(np.uint8))

        if self.args.save_attention_images:
            for h in attn_hook_handles:
                h.remove()
        if self.args.save_alignment_images:
            for h in alignment_hook_handles:
                h.remove()

        loc_f1_score = self.total_evaluator_loc.Pixel_F1_score()
        damage_f1_score: np.ndarray = self.total_evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

        # Make the scores more readable
        loc_f1_score     = float(np.round(loc_f1_score     * 100, 4))
        harmonic_mean_f1 = float(np.round(harmonic_mean_f1 * 100, 4))
        oaf1             = float(np.round(oaf1             * 100, 4))
        for i in range(len(damage_f1_score)): damage_f1_score[i] = np.round(damage_f1_score[i] * 100, 4)

        # print the confusion matrices
        conf_loc_count = np.array(self.total_evaluator_loc.confusion_matrix, dtype=np.int64)
        conf_clf_count = np.array(self.total_evaluator_clf.confusion_matrix, dtype=np.int64)
        conf_loc_norm = conf_loc_count / conf_loc_count.astype(np.float64).sum(axis=1, keepdims=True)
        conf_clf_norm = conf_clf_count / conf_clf_count.astype(np.float64).sum(axis=1, keepdims=True)
        logging.info(f"Confusion Matrix of Localization:\n{conf_loc_count}")
        logging.info(f"Confusion Matrix of Localization - Normalized:\n{conf_loc_norm}")
        logging.info(f"Confusion Matrix of Classification:\n{conf_clf_count}")
        logging.info(f"Confusion Matrix of Classification - Normalized:\n{conf_clf_norm}")

        logging.info(f'lofF1 is {loc_f1_score:.4f}, clfF1 is {harmonic_mean_f1:.4f}, oaF1 is {oaf1:.4f}, sub class F1 score is {damage_f1_score}')


def main():
    parser = argparse.ArgumentParser(description="Inference on Building Damage Assessment (xBD, mwBTFreddy, ...)")
    parser.add_argument('--cfg', type=str, default='/home/songjian/project/MambaCD/VMamba/classification/configs/vssm1/vssm_base_224.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--pretrained_weight_path', type=str)
    parser.add_argument('--dataset', type=str, default='xBD')
    parser.add_argument('--type', type=str, default='train')
    parser.add_argument('--test_dataset_path', type=str, default='/home/songjian/project/datasets/SYSU/test')
    parser.add_argument('--test_data_list_path', type=str, default='/home/songjian/project/datasets/SYSU/test_list.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='MambaBDA_Tiny')
    parser.add_argument('--result_saved_path', type=str, default='../results')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    parser.add_argument('--logfile', type=str, help="full path to log file")
    parser.add_argument('--save_output_images', type=bool, action=argparse.BooleanOptionalAction, default=True) # type "--no-save_output_images" to set to False
    parser.add_argument('--extension', type=str, help='dataset image file extension without dot ("png", "tif", etc.)')
    parser.add_argument('--enable_alignment', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_attn_gate_building', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_attn_gate_damage', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--save_attention_images', type=bool, action=argparse.BooleanOptionalAction, default=True) # type "--no-save_attention_images" to set to False
    parser.add_argument('--save_alignment_images', type=bool, action=argparse.BooleanOptionalAction, default=True) # type "--no-save_alignment_images" to set to False
    parser.add_argument('--measure_efficiency', type=bool, action=argparse.BooleanOptionalAction, default=True) # type "--no-measure_efficiency" to set to False

    args = parser.parse_args()

    #*-- LOGGING INIT
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name: str = args.model_type
    if args.logfile is None:
        print(" !! WARNING !! Log file parameter is empty, using default name for log file.")
        logfile_path = f"/storage/alperengenc/change_detection/ChangeMamba_AG/LOGLAR_CMAG/infer_{now}_{model_name}.log"
    else:
        logfile_path = args.logfile
    logging.basicConfig(
        level=logging.INFO,  # INFO / DEBUG
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(logfile_path, mode="a"), # log to file
            logging.StreamHandler() # log to stdout
        ]
    )
    logging.info(f"MAIN - START")
    logging.info(f" > ALINGNMENT set to {args.enable_alignment}")
    logging.info(f" > ATTENTION GATE set to -> Building: {args.enable_attn_gate_building}, Damage: {args.enable_attn_gate_damage}")

    args_copy = copy.deepcopy(vars(args))
    args_pretty = json.dumps(args_copy, indent=4)
    logging.info(f"Command Line Args:\n{args_pretty}")

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    trainer = Trainer(args)
    trainer.infer()


if __name__ == "__main__":
    try:
        main()
        logging.info(f"MAIN - DONE.")
    except Exception as exc:
        logging.info(f"MAIN - ERROR: {exc}", exc_info=True, stack_info=True)
    finally:
        logging.info(f"MAIN - EXIT.")
