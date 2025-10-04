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
from changedetection.datasets.make_data_loader import ChangeDetectionDatset, make_data_loader, DamageAssessmentDatset
from changedetection.utils_func.metrics import Evaluator
from changedetection.models.ChangeMambaBDA import ChangeMambaBDA

import changedetection.utils_func.lovasz_loss as L

import logging
import json
import copy
from typing import Sequence
import random
from collections import deque

from changedetection.models.alignment_module import AlignmentArgs
from changedetection.models.attn_gate import AttentionGateArgs


def set_deterministic_seed(seed: int):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For reproducibility in cudnn
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Ensures CUDA uses deterministic algorithms where possible
    torch.use_deterministic_algorithms(True)

    # Optional: Make hash-based ops deterministic (Python 3.3+)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # to solve cublas error, recommended solution: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


class FocalLossCE(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Sequence[float] | torch.Tensor | None = None, ignore_index: int = 255):
        super().__init__()
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        if alpha is not None:
            # len(alpha) == C' (here 4 classes: [1,2,3,4])
            alpha_tensor = torch.as_tensor(alpha, dtype=torch.float32)
            #* register_buffer keeps tensor on correct device but not a trainable parameter
            #* this creates self.alpha: torch.Tensor. For type hinting, type is written in the else block.
            self.register_buffer("alpha", alpha_tensor)
        else:
            self.alpha: torch.Tensor | None = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Expects `logits: [B,C,H,W]`, `targets: [B,H,W]`.
        Works with `B=1` as usual. If a single image is passed without batch, first `unsqueeze(0)` should be called.
        """
        valid = (targets != self.ignore_index)
        if not valid.any():
            return torch.zeros((), device=logits.device)
        
        # logits: [B,C,H,W], targets: [B,H,W] with {1..C', 255}
        # 1) CE per pixel (ignored pixels produce 0 loss and 0 grad)
        ce = F.cross_entropy(logits, targets, reduction="none", ignore_index=self.ignore_index) # [B,H,W]

        # 2) pt = exp(-CE) => probabilities
        with torch.no_grad():
            pt = torch.exp(-ce)  # [B,H,W] in (0,1]

        focal = (1.0 - pt) ** self.gamma * ce  # [B,H,W]

        # 3) class weighting α (per true class)
        if self.alpha is not None:
            device = logits.device
            a = self.alpha.to(device=device)

            # Build safe class indices for gather
            t_safe = targets.clone().to(device)
            t_safe[~valid] = 1  # any valid class id in [1..C']; here choose 1
            # Map class ids {1..C'} -> alpha index {0..C'-1}

            # class ids are {1,2,3,4}; convert to 0-based indices {0,1,2,3}
            idx = (t_safe - 1).view(-1) # [B*H*W]

            alpha_per_pix = a.index_select(0, idx) # [B*H*W]
            alpha_per_pix = alpha_per_pix.view_as(t_safe).to(focal.dtype) # [B,H,W]

            focal = focal * alpha_per_pix # [B,H,W]

        # 4) mean over valid pixels only
        focal = focal[valid]
        return focal.mean()


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.train_data_loader = make_data_loader(args)

        if self.args.measure_train_scores:
            TRAIN_BUF_MAXLEN = 1024 # number of batches (not batch size)
            self.train_buf: deque[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = deque(maxlen=TRAIN_BUF_MAXLEN) 
            """each element in the train_buf will be a `tuple[preds_loc, preds_clf, labels_loc, labels_clf]` (all tensors of shape [B,H,W])"""
            self.train_evaluator_loc = Evaluator(num_class=2)
            self.train_evaluator_clf = Evaluator(num_class=5)
            logging.info(f" > TRAIN EVALUATION params: {TRAIN_BUF_MAXLEN = }")
        else:
            logging.info(f" > TRAIN EVALUATION disabled.")


        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)

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
        # self.model_save_path = os.path.join(args.model_param_path, args.dataset,
        #                                     args.model_type + '_' + str(time.time()))
        self.model_save_path = args.model_param_path
        self.lr = args.learning_rate
        self.epoch = args.max_iters // args.batch_size

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

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

        self.optim = optim.AdamW(self.deep_model.parameters(),
                                 lr=args.learning_rate,
                                 weight_decay=args.weight_decay)
        if self.args.focal_loss:
            # alpha = [1.0, 2.3, 1.3, 1.1] # high priority to minor damage
            alpha = [0.6, 1.6, 1.1, 1.1] # closer to inverse frequencies
            gamma = 1.5
            logging.info(f" > FOCAL LOSS params: {alpha = }, {gamma = }")
            self.focal_loss_func = FocalLossCE(gamma=gamma, alpha=alpha, ignore_index=255)


    def training(self):
        print('---------starting training-----------')
        logging.log(logging.INFO, '---------starting training-----------')

        best_kc = 0.0
        best_round = []
        torch.cuda.empty_cache()
        elem_num = len(self.train_data_loader)
        train_enumerator = enumerate(self.train_data_loader)

        number_of_validations = self.args.validations
        VAL_STEP = max(1, elem_num // number_of_validations)
        logging.log(logging.INFO, f"{VAL_STEP=}, ({number_of_validations = })")

        skipped_count = 0
        valid_results: dict[int, tuple] = {} # key is iteration (step), value is whole validation result scores.
        train_results: dict[int, tuple] = {} # key is iteration (step), value is whole validation result scores.

        for _ in tqdm(range(elem_num)):
            itera, data = train_enumerator.__next__()
            pre_change_imgs, post_change_imgs, labels_loc, labels_clf, data_name = data

            pre_change_imgs = pre_change_imgs.cuda()
            post_change_imgs = post_change_imgs.cuda()
            labels_loc = labels_loc.cuda().long()
            labels_clf = labels_clf.cuda().long()

            valid_labels_clf = (labels_clf != 255).any()
            if not valid_labels_clf:
               skipped_count += 1
            #    logging.info(f"skipped step {itera} (total: {skipped_count})")
               continue
            
            output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

            if self.args.measure_train_scores:
                with torch.no_grad():
                    pred_loc: torch.Tensor = output_loc.argmax(1) # [B,H,W]
                    pred_clf: torch.Tensor = output_clf.argmax(1)
                    self.train_buf.append((
                        pred_loc.detach().cpu(), 
                        pred_clf.detach().cpu(),
                        labels_loc.detach().cpu(),
                        labels_clf.detach().cpu(),
                    ))


            self.optim.zero_grad()

            ce_loss_loc = F.cross_entropy(output_loc, labels_loc, ignore_index=255)
            lovasz_loss_loc = L.lovasz_softmax(F.softmax(output_loc, dim=1), labels_loc, ignore=255)

            if self.args.focal_loss:
                # hybrid CE and FOCAL combination
                ce_plain = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
                ce_focal = self.focal_loss_func(output_clf, labels_clf)
                ce_loss_clf = 0.5 * ce_plain + 0.5 * ce_focal
            else:
                ce_loss_clf = F.cross_entropy(output_clf, labels_clf, ignore_index=255)
            lovasz_loss_clf = L.lovasz_softmax(F.softmax(output_clf, dim=1), labels_clf, ignore=255)

            final_loss = ce_loss_loc + ce_loss_clf + (0.5 * lovasz_loss_loc + 0.75 * lovasz_loss_clf)

            final_loss.backward()

            self.optim.step()

            if (itera + 1) % 50 == 0:
                log = f'iter is {itera + 1} / {elem_num} [skipped {skipped_count:>4}] | loc. loss = {ce_loss_loc + lovasz_loss_loc :<.10f}, classif. loss = {ce_loss_clf + lovasz_loss_clf :<.10f}'
                print(log)
                logging.log(logging.INFO, log)
            is_last_step = (itera + 1 >= elem_num)
            if (itera + 1) % VAL_STEP == 0 and not is_last_step: # do not start validation in the last step, final validation will be started after training ends.
                try:
                    self.deep_model.eval()
                    loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score = self.validation()
                    valid_results[itera+1] = loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score
                    if oaf1 > best_kc:
                        model_save_path = os.path.join(self.model_save_path, f'model_step{itera + 1}.pth')
                        torch.save(self.deep_model.state_dict(), model_save_path)
                        logging.info(f"Model saved in: {model_save_path}")
                        best_kc = oaf1
                        best_round = [loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score]
                    
                    if self.args.measure_train_scores:
                        tr_locf1, tr_clff1, tr_oaf1, tr_dmgs = self.train_buffer_metrics()
                        train_results[itera+1] = tr_locf1, tr_clff1, tr_oaf1, tr_dmgs

                except Exception as exc:
                    logging.error(f"VALIDATION - ERRROR: {exc}", exc_info=True, stack_info=True)
                finally:
                    self.deep_model.train()


        log = "-----------Training is completed-----------"
        print(log)
        logging.log(logging.INFO, log)

        #* After training is complete, save the model regardless of the best score (the final version after training should be stored, while validating it may not have been saved due to errors etc.)
        model_save_path = os.path.join(self.model_save_path, f'model_step{itera + 1}_last.pth')
        torch.save(self.deep_model.state_dict(), model_save_path)
        logging.info(f"Model saved in: {model_save_path}")

        logging.info(f"!! Total Skipped: {skipped_count} ({skipped_count/elem_num * 100:.2f}%)")

        #*-- Validation after training
        self.deep_model.eval()
        loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score = self.validation()
        valid_results[-1] = loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score # -1 means after training is completed.
        log = f"{loc_f1_score=}, {harmonic_mean_f1=}, {oaf1=}, {damage_f1_score=}"
        print(log)
        logging.log(logging.INFO, log)

        if oaf1 > best_kc:
            #* model already saved above, after training is completed, so do not save here.
            # model_save_path = os.path.join(self.model_save_path, f'model_step{itera + 1}.pth')
            # torch.save(self.deep_model.state_dict(), model_save_path)
            # logging.info(f"Model saved in: {model_save_path}")
            best_kc = oaf1
            best_round = [loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score]

        if self.args.measure_train_scores:
            tr_locf1, tr_clff1, tr_oaf1, tr_dmgs = self.train_buffer_metrics()
            train_results[-1] = tr_locf1, tr_clff1, tr_oaf1, tr_dmgs
        
        self.deep_model.train()

        logging.info("Validation Results:")
        for step, scores in valid_results.items():
            logging.info(f"[TEST ] Step {step:>5}: {scores}")
            if self.args.measure_train_scores:
                logging.info(f"[TRAIN] Step {step:>5}: {train_results[step]}\n")

        print('The accuracy of the best round is ', best_round)
        logging.log(logging.INFO, f'The accuracy of the best round is: {best_round}')


    def train_buffer_metrics(self):
        logging.info('---------starting train set evaluation-----------')
        if len(self.train_buf) == 0:
            logging.info("Train buffer empty, returning 0.")
            return 0.0, 0.0, 0.0, np.zeros((4,), dtype=np.float32)  # safe default
        
        self.train_evaluator_loc.reset()
        self.train_evaluator_clf.reset()
        with torch.no_grad():
            for (preds_loc, preds_clf, labels_loc, labels_clf) in list(self.train_buf):
                preds_loc  = preds_loc.numpy()
                labels_loc = labels_loc.numpy()

                preds_clf  = preds_clf.numpy()[labels_loc > 0]
                labels_clf = labels_clf.numpy()[labels_loc > 0]

                self.train_evaluator_loc.add_batch(labels_loc, preds_loc)
                self.train_evaluator_clf.add_batch(labels_clf, preds_clf)

        loc_f1_score = self.train_evaluator_loc.Pixel_F1_score()
        damage_f1_score: np.ndarray = self.train_evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

        # Make the scores more readable
        loc_f1_score     = np.round(loc_f1_score     * 100, 4)
        harmonic_mean_f1 = np.round(harmonic_mean_f1 * 100, 4)
        oaf1             = np.round(oaf1             * 100, 4)
        for i in range(len(damage_f1_score)): damage_f1_score[i] = np.round(damage_f1_score[i] * 100, 4)

        logging.info(f'[TrainBuf] locF1 is {loc_f1_score:.4f}, clfF1 is {harmonic_mean_f1:.4f}, oaF1 is {oaf1:.4f}, sub class F1 score is {damage_f1_score}')
        return loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score


    def validation(self):
        print('---------starting evaluation-----------')
        logging.log(logging.INFO, '---------starting evaluation-----------')
        self.evaluator_loc.reset()
        self.evaluator_clf.reset()
        if self.args.extension is None:
            ext = "tif" if 'mwBTFreddy' in self.args.dataset else "png"
        else: 
            ext = self.args.extension
        dataset = DamageAssessmentDatset(self.args.test_dataset_path, self.args.test_data_name_list, 256, None, 'test', extension=ext)
        val_data_loader = DataLoader(dataset, batch_size=1, num_workers=4, drop_last=False)
        torch.cuda.empty_cache()

        # vbar = tqdm(val_data_loader, ncols=50)
        with torch.no_grad():
            for itera, data in enumerate(val_data_loader):
                if itera % 100 == 0:
                    log = f'validation: {itera:>4}/{len(val_data_loader):>4} ({datetime.now():%Y-%m-%d_%H-%M-%S})'
                    print(log)
                    logging.log(logging.INFO, log)

                pre_change_imgs, post_change_imgs, labels_loc, labels_clf, _ = data

                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                labels_loc = labels_loc.cuda().long()
                labels_clf = labels_clf.cuda().long()


                # input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)
                output_loc, output_clf = self.deep_model(pre_change_imgs, post_change_imgs)

                output_loc = output_loc.data.cpu().numpy()
                output_loc = np.argmax(output_loc, axis=1)
                labels_loc = labels_loc.cpu().numpy()

                output_clf = output_clf.data.cpu().numpy()
                output_clf = np.argmax(output_clf, axis=1)
                labels_clf = labels_clf.cpu().numpy()

                self.evaluator_loc.add_batch(labels_loc, output_loc)
                
                output_clf = output_clf[labels_loc > 0]
                labels_clf = labels_clf[labels_loc > 0]
                self.evaluator_clf.add_batch(labels_clf, output_clf)

        loc_f1_score = self.evaluator_loc.Pixel_F1_score()
        damage_f1_score: np.ndarray = self.evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1

        # Make the scores more readable
        loc_f1_score     = np.round(loc_f1_score     * 100, 4)
        harmonic_mean_f1 = np.round(harmonic_mean_f1 * 100, 4)
        oaf1             = np.round(oaf1             * 100, 4)
        for i in range(len(damage_f1_score)): damage_f1_score[i] = np.round(damage_f1_score[i] * 100, 4)

        # print the confusion matrices
        conf_loc_count = np.array(self.evaluator_loc.confusion_matrix, dtype=np.int64)
        conf_clf_count = np.array(self.evaluator_clf.confusion_matrix, dtype=np.int64)
        conf_loc_norm = conf_loc_count / conf_loc_count.astype(np.float64).sum(axis=1, keepdims=True)
        conf_clf_norm = conf_clf_count / conf_clf_count.astype(np.float64).sum(axis=1, keepdims=True)
        logging.info(f"Confusion Matrix of Localization:\n{conf_loc_count}")
        logging.info(f"Confusion Matrix of Localization - Normalized:\n{conf_loc_norm}")
        logging.info(f"Confusion Matrix of Classification:\n{conf_clf_count}")
        logging.info(f"Confusion Matrix of Classification - Normalized:\n{conf_clf_norm}")

        logging.info(f'lofF1 is {loc_f1_score:.4f}, clfF1 is {harmonic_mean_f1:.4f}, oaF1 is {oaf1:.4f}, sub class F1 score is {damage_f1_score}')
        return loc_f1_score, harmonic_mean_f1, oaf1, damage_f1_score


def main():
    parser = argparse.ArgumentParser(description="Training on Building Damage Assessment (xBD, mwBTFreddy, ...)")
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
    parser.add_argument('--train_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/train')
    parser.add_argument('--train_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/train_all.txt')
    parser.add_argument('--test_dataset_path', type=str, default='/data/ggeoinfo/datasets/xBD/test')
    parser.add_argument('--test_data_list_path', type=str, default='/data/ggeoinfo/datasets/xBD/xBD_list/val_all.txt')
    parser.add_argument('--shuffle', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--train_data_name_list', type=list)
    parser.add_argument('--test_data_name_list', type=list)
    parser.add_argument('--start_iter', type=int, default=0)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--max_iters', type=int, default=240000)
    parser.add_argument('--model_type', type=str, default='ChangeMambaBDA')
    parser.add_argument('--model_param_path', type=str, default='../saved_models')

    parser.add_argument('--resume', type=str)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-3)

    parser.add_argument('--logfile', type=str, help="full path to log file")
    parser.add_argument('--extension', type=str, help='dataset image file extension without dot ("png", "tif", etc.)')
    parser.add_argument('--focal_loss', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_alignment', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_attn_gate_building', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--enable_attn_gate_damage', type=bool, action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--deterministic', type=bool, action=argparse.BooleanOptionalAction, default=False, help="(can't be used for now (2025.09.29, torch==2.5.0) because of non-deterministic functions (e.g. F.cross_entropy))")
    parser.add_argument('--validations', type=int, default=8)
    parser.add_argument('--measure_train_scores', type=bool, action=argparse.BooleanOptionalAction, default=False)

    args = parser.parse_args()
    with open(args.train_data_list_path, "r") as f:
        # data_name_list = f.read()
        data_name_list = [data_name.strip() for data_name in f]
    args.train_data_name_list = data_name_list

    with open(args.test_data_list_path, "r") as f:
        # data_name_list = f.read()
        test_data_name_list = [data_name.strip() for data_name in f]
    args.test_data_name_list = test_data_name_list

    #*-- LOGGING INIT
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name: str = args.model_type
    if args.logfile is None:
        print(" !! WARNING !! Log file parameter is empty, using default name for log file.")
        logfile_path = f"/storage/alperengenc/change_detection/ChangeMamba_AG/LOGLAR_CMAG/train_{now}_{model_name}.log" # TODO get from args
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
    logging.log(logging.INFO, f"MAIN - START")
    logging.info(f" > FOCAL LOSS set to {args.focal_loss}")
    logging.info(f" > ALINGNMENT set to {args.enable_alignment}")
    logging.info(f" > ATTENTION GATE set to -> Building: {args.enable_attn_gate_building}, Damage: {args.enable_attn_gate_damage}")

    args_copy = copy.deepcopy(vars(args))
    args_copy.pop("train_data_name_list")
    args_copy.pop("test_data_name_list")
    args_pretty = json.dumps(args_copy, indent=4)
    logging.log(logging.INFO, f"Command Line Args:\n{args_pretty}")

    if args.deterministic:
        seed = 2025
        logging.info(f"Starting in DETERMINISTIC mode ({seed = }).")
        set_deterministic_seed(seed)
    else:
        logging.info(f"Starting in RANDOM mode / not deterministic.")


    trainer = Trainer(args)
    trainer.training()


if __name__ == "__main__":
    try:
        main()
        logging.log(logging.INFO, f"MAIN - DONE.")
    except Exception as exc:
        logging.log(logging.ERROR, f"MAIN - ERROR: {exc}", exc_info=True, stack_info=True)
    finally:
        logging.log(logging.INFO, f"MAIN - EXIT.")
