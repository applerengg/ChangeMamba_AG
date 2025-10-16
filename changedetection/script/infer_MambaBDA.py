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

from changedetection.models.alignment_module import AlignmentArgs
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


def save_all_attn_maps(pre_mask, post_mask, attn_maps: dict[str, torch.Tensor], out_path: str):
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

    fig, axs = plt.subplots(2, 4, figsize=(16, 8))

    def get_heatmap(attn_tensor: torch.Tensor) -> np.ndarray:
        heat = F.interpolate(attn_tensor, size=(H, W), mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
        return heat
    
    def display_heatmaps(attn_maps: dict, row_idx: int, mask: torch.Tensor, mask_title: str):
        # Column 0: GT mask
        axs[row_idx, 0].imshow(mask, cmap="gray")
        axs[row_idx, 0].set_title(mask_title, fontsize=10)
        axs[row_idx, 0].axis("off")
        for col_idx, (name, attn_tensor) in enumerate(attn_maps.items(), start=1):
            heat = get_heatmap(attn_tensor)
            axs[row_idx, col_idx].imshow(mask, cmap="gray")
            axs[row_idx, col_idx].imshow(heat, cmap="jet", alpha=0.6)
            axs[row_idx, col_idx].set_title(f"{name} on {mask_title}")
            axs[row_idx, col_idx].axis("off")

    if pre_mask is not None:
        display_heatmaps(building_maps, row_idx=0, mask=pre_mask, mask_title="Building GT")

    if post_mask is not None:
        display_heatmaps(damage_maps, row_idx=1, mask=post_mask, mask_title="Damage GT")

    plt.tight_layout()
    plt.savefig(f"{out_path}", dpi=150)
    plt.close(fig)
    logging.info(f"Saved attention visualization: {out_path}")



class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)
        self.total_evaluator_loc = Evaluator(num_class=2)
        self.total_evaluator_clf = Evaluator(num_class=5)

        if args.enable_alignment:
            alignment_args = AlignmentArgs(enabled=True, stages=(2,), mid_ch=64)
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

        if not os.path.exists(self.building_map_T1_saved_path):
            os.makedirs(self.building_map_T1_saved_path)
        if not os.path.exists(self.change_map_T2_saved_path):
            os.makedirs(self.change_map_T2_saved_path)
        if not os.path.exists(self.attention_map_saved_path):
            os.makedirs(self.attention_map_saved_path)


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
            attn_maps, handles = register_attn_hooks(self.deep_model)

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

                # --- visualize first AG map for this sample ---
                building_available = labels_loc.max().item() != 0
                if not building_available:
                    logging.info(f" > No building in {names[0]}, skipping attention visualization.")
                if self.args.save_attention_images and len(attn_maps) > 0 and building_available:
                    # img = denormalize_img(pre_change_imgs[0], mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    pre_mask = None
                    post_mask = None
                    # save_all_attn_maps(img, mask, attn_maps, os.path.join(self.attention_map_saved_path, f"{names[0]}_all.png"))
                    if self.args.enable_attn_gate_building:
                        pre_mask = labels_loc[0].detach().cpu().numpy()
                    if self.args.enable_attn_gate_damage:
                        post_mask = labels_clf[0].detach().cpu().numpy()
                        post_mask[post_mask == 255] = 0
                    save_all_attn_maps(pre_mask, post_mask, attn_maps, os.path.join(self.attention_map_saved_path, f"{names[0]}_attentions.png"))
                    # if itera > 10:
                    #     break  # DEBUG (quick results, only visualize first n samples)

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

                if self.args.save_output_images:
                    image_name = names[0] + '.png'

                    output_loc = np.squeeze(output_loc)
                    output_loc[output_loc > 0] = 255

                    output_clf = map_labels_to_colors(np.squeeze(output_clf), ori_label_value_dict=ori_label_value_dict, target_label_value_dict=target_label_value_dict)
                    output_clf[output_loc == 0] = 0

                    imageio.imwrite(os.path.join(self.building_map_T1_saved_path, image_name), output_loc.astype(np.uint8))
                    imageio.imwrite(os.path.join(self.change_map_T2_saved_path, image_name), output_clf.astype(np.uint8))

        for h in handles:
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
