import glob
import os
from pathlib import Path
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

train_list = glob.glob("./xBD_complete_png/train_combined/images/*.png")
train_list = [Path(p).stem for p in train_list]
with open("./xBD_complete_png/train_combined_list.txt", "w") as f:
    for item in tqdm(train_list):
        f.write(f"{item}\n")

test_list = glob.glob("./xBD_complete_png/test/images/*.png")
test_list = [Path(p).stem for p in test_list]
with open("./xBD_complete_png/test_list.txt", "w") as f:
    for item in tqdm(test_list):
        f.write(f"{item}\n")
