import sys
# sys.path.append('/home/songjian/project/MambaCD')
sys.path.append("/storage/alperengenc/change_detection/ChangeMamba_AG/")

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


class Trainer(object):
    def __init__(self, args):
        self.args = args
        config = get_config(args)

        self.evaluator_loc = Evaluator(num_class=2)
        self.evaluator_clf = Evaluator(num_class=5)
        self.total_evaluator_loc = Evaluator(num_class=2)
        self.total_evaluator_clf = Evaluator(num_class=5)

        self.deep_model = ChangeMambaBDA(
            output_building=2, output_damage=5,
            pretrained=args.pretrained_weight_path,
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

        if not os.path.exists(self.building_map_T1_saved_path):
            os.makedirs(self.building_map_T1_saved_path)
        if not os.path.exists(self.change_map_T2_saved_path):
            os.makedirs(self.change_map_T2_saved_path)


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

        loc_f1_score = self.total_evaluator_loc.Pixel_F1_score()
        damage_f1_score = self.total_evaluator_clf.Damage_F1_socore()
        harmonic_mean_f1 = len(damage_f1_score) / np.sum(1.0 / damage_f1_score)
        oaf1 = 0.3 * loc_f1_score + 0.7 * harmonic_mean_f1
        logging.info(f'lofF1 is {loc_f1_score}, clfF1 is {harmonic_mean_f1}, oaF1 is {oaf1}, sub class F1 score is {damage_f1_score}')


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
