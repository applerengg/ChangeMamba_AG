import glob
import os
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

BASE_PATH = "/mnt/storage1/alpgenc/change_detection/datasets/earthquake_turkey/earthquake-turkey"

################################################################################################################

all_images = glob.glob(f"{BASE_PATH}/images/*.png")

#* Create a list that contains all images
all_list = [Path(p).stem for p in all_images]
with open(f"{BASE_PATH}/all_list.txt", "w") as f:
    for item in tqdm(all_list):
        f.write(f"{item}\n")

print("Created 'all_list.txt'")

################################################################################################################

#* create train_list.txt and test_list.txt

@dataclass
class DataPaths:
    pre_img: str
    post_img: str
    pre_target: str
    post_target: str
    stem: str
    """base name; without path, pre-post suffix and extension."""

all_pre_images = glob.glob(f"{BASE_PATH}/images/*_pre_disaster.png")
all_data: list[DataPaths] = []

for i, pre_img_path in tqdm(enumerate(all_pre_images)):
    post_img_path = pre_img_path.replace("pre", "post")
    img_base, _, _ = pre_img_path.rsplit("_", 2) # 

    target_base = img_base.replace("images", "targets")
    pre_target_path = f"{target_base}_pre_disaster_target.png"
    post_target_path = f"{target_base}_post_disaster_target.png"

    stem = os.path.basename(img_base)

    all_data.append(DataPaths(pre_img_path, post_img_path, pre_target_path, post_target_path, stem))

print("Generated path info")

TOTAL = len(all_data)
TRAIN_RATIO = 0.80
BORDER_IDX = round(TOTAL * TRAIN_RATIO)

with open(f"{BASE_PATH}/train_list.txt", "w") as f:
    for i in range(0, BORDER_IDX):
        d = all_data[i]
        f.write(f"{d.stem}_pre_disaster\n")
        f.write(f"{d.stem}_post_disaster\n")
print("Generated 'train_list.txt'")

with open(f"{BASE_PATH}/test_list.txt", "w") as f:
    for i in range(BORDER_IDX, TOTAL):
        d = all_data[i]
        f.write(f"{d.stem}_pre_disaster\n")
        f.write(f"{d.stem}_post_disaster\n")
print("Generated 'test_list.txt'")
