"""
Generalized version of xbd_dataset_create_targets.py script.
Works for both xBD and mwBTFreddy.
"""
##################################################################################################################
# IMPORTANT NOTE (2025.08.07): cv2.fillPoly outputs vary slightly depending on the cv2 (maybe numpy) versions.
# env: changemamba2, python: 3.10.15, opencv-python: 4.10.0.84, numpy: 2.1.2 --> max error: 2.732%
# env: vision      , python: 3.13.5 , opencv-python: 4.12.0   , numpy: 2.3.2 --> max error: 0.142%
##################################################################################################################

import os
import glob
import json
import numpy as np
from PIL import Image
from shapely import wkt
from matplotlib import pyplot as plt
import cv2
from pprint import pprint
import argparse


DAMAGE_MAPPING = {
    "no-damage": 1,
    "minor-damage": 2,
    "major-damage": 3,
    "destroyed": 4,

    # TODO: check with the un-classified type. can be 0 or 1. in my experiments I saw that the original target files interpret it as 0.
    "un-classified": 0, # [2025.08.07] can be buildings such as obstructed by clouds, therefore used as not-building (0).
}

UNKNOWN_TYPES = set()

def create_target_mask_from_label(label_json_path: str, out_shape=(1024, 1024), verbose: bool = False) -> np.ndarray:
    with open(label_json_path, 'r') as f:
        label_data = json.load(f)

    mask = np.zeros(out_shape, dtype=np.uint8)
    xy_features = label_data['features'].get("xy", [])
    if not xy_features:
        # print(f"No xy features found in labels.json. Returning empty mask. (lng_lat: {label_data['features'].get('lng_lat', [])})")
        return mask

    for xy_feat in xy_features:
        props = xy_feat['properties']
        if props.get("feature_type") != "building":
            ## do not include non-building features (in my experiments, only building features were present but still added as a measure).
            # print(f"Feature type is not building: {props.get('feature_type')}. Skipping the feature.")
            continue
        damage_type = props.get('subtype')
        if damage_type not in DAMAGE_MAPPING:
            # print(f"Unknown damage type: {damage_type} in {label_json_path}. Skipping the feature.")
            UNKNOWN_TYPES.add(damage_type)
        damage_level = DAMAGE_MAPPING.get(damage_type, 1)  # Default to 1 (building no-damage) if type is not found.
        
        if damage_level > 1 and "pre_disaster" in label_json_path:
            if verbose:
                print(" !! Pre Disaster labels are localization labels and must not contain damage levels.")
                print(f" !! {label_json_path} contains damage level {damage_level}, setting to 1.")
            damage_level = 1
        
        polygon = wkt.loads(xy_feat['wkt'])
        pxcoords = np.array(polygon.exterior.coords)
        pxcoords = np.round(pxcoords).astype(np.int32)
        cv2.fillPoly(mask, [pxcoords], damage_level)

    return mask


def check_target_error(generated_target: np.ndarray, expected_target: np.ndarray, verbose: bool = False):
    diff_img = np.where(expected_target != generated_target, 1, 0)
    incorrect_pixel_count = diff_img.sum()
    ground_truth_non_zero = np.where(expected_target > 0, 1, 0).sum()
    generated_mask_non_zero = np.where(generated_target > 0, 1, 0).sum()
    if ground_truth_non_zero == 0:
        if generated_mask_non_zero == 0:
            incorrect_pixel_percentage = 0.0
        else:
            ## can be interpreted as 100% error or error rate can be calculated using image size.
            incorrect_pixel_percentage = 100.0
    else:
        incorrect_pixel_percentage = incorrect_pixel_count / ground_truth_non_zero * 100
    if verbose:
        print("Ground truth non-zero:", ground_truth_non_zero) # count of non-zero pixels in ground truth
        print("Created mask non-zero:", generated_mask_non_zero) # count of non-zero pixels in created mask
        print(f"Incorrect pixel count: {incorrect_pixel_count} ({incorrect_pixel_percentage: .3f}%)", ) # count of pixels that are different
    return incorrect_pixel_percentage

##################################################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels_dir', type=str, required=True)
    parser.add_argument('--out_folder_name', type=str, default="targets2", help="Name of the folder to save the output mask, which must be in the same dir as labels folder. Must be already existing empty folder.")
    parser.add_argument('--verbose', type=bool, action=argparse.BooleanOptionalAction, default=True) # --no-verbose for False
    parser.add_argument('--check_error', type=bool, action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    print(args)
    print(args.check_error)

    ## Configuration
    labels_dir: str = args.labels_dir # xBD: "./xBD_complete_png/train/labels", mwBTFreddy: "./mwBTFreddy_dataset/mwBTFreddy_v1.0/labels"
    out_folder_name: str = args.out_folder_name 
    check_error: bool = args.check_error  # if there is a reference ground truth, Set to True to check for errors in target mask creation
    verbose: bool = args.verbose  # Set to True for detailed output
    
    successful_error_threshold1 = 0.1  # Percentage of incorrect pixels allowed
    successful_error_threshold2 = 1.0  # Percentage of incorrect pixels allowed

    ## change directory to the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f"Current working directory: {os.getcwd()}")

    label_paths = glob.glob(f"{labels_dir}/*.json")
    label_count = len(label_paths)

    very_successful_count = 0
    successful_count = 0
    errors_info = []
    for i, label_path in enumerate(label_paths):
        out_path = label_path.replace('labels', out_folder_name).replace('.json', '_target.png')
        res = create_target_mask_from_label(label_path, verbose=verbose)
        
        ## IMPORTANT: FOLLOWING LINE SAVES THE GENERATED MASK. COMMENT OUT FOR TESTING, DOUBLE CHECK OUTPUT PATH BEFORE RUNNING.
        Image.fromarray(res).save(out_path)

        if check_error:
            expected_target_img = cv2.imread(out_path.replace('targets2', 'targets'), cv2.IMREAD_GRAYSCALE)
            err_rate = check_target_error(res, expected_target_img, verbose)
            print(f"[{i:>5} / {label_count}] Error rate: {err_rate:.3f}% ({label_path})")
            if err_rate <= successful_error_threshold1:
                very_successful_count += 1
            elif err_rate <= successful_error_threshold2:
                successful_count += 1
            
            if err_rate > 0: # only add to errors_info if there is a non-zero error
                errors_info.append((i, label_path, err_rate))
        else:
            print(f"[{i:>5} / {label_count}] Successfully created target mask for {label_path}")


    pprint(errors_info)
    # print(f"{errors_info =}")
    print(f"> Total Very Successful: {very_successful_count}")
    print(f"> Total Successful: {successful_count}")
    print(f"> # Nonzero Errors: {len(errors_info)}")
    print(f"> MAX Error: {max(errors_info, key=lambda x: x[2]) if errors_info else 'N/A'}")
    print(f"{UNKNOWN_TYPES = }")

if __name__ == "__main__":
    main()
