import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from tqdm import tqdm

# Target output size (what user wants)
TARGET_SIZE = 64
# Processing size (must be divisible by 64 for UNet compatibility: 64/8=8 latent)
PROCESS_SIZE = 64


def pad_to_size(img, target_size, pad_value=0):
    """Pad image to target_size with center alignment"""
    h, w = img.shape[:2]
    pad_h = target_size - h
    pad_w = target_size - w
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    if len(img.shape) == 3:
        return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode='constant', constant_values=pad_value)
    return np.pad(img, ((top, bottom), (left, right)), mode='constant', constant_values=pad_value)


class MyDataset(Dataset):
    def __init__(self, root_dir='/root/autodl-tmp/dreamnav/', dataset_type='try_train'):
        self.data = []
        self.root = os.path.join(root_dir, dataset_type) + '/'
        for building_id in tqdm(os.listdir(self.root)):
            now_dir_path = self.root + building_id + '/'
            for item_id in os.listdir(now_dir_path):
                json_path = now_dir_path + item_id
                with open(json_path, "r", encoding="utf-8") as f:
                    now_json = json.load(f)
                image_patha = root_dir + '/tours/' + now_json["image_a"]
                image_pathb = root_dir + '/tours/' + now_json["image_b"]
                self.data.append([image_patha, image_pathb, now_json["heading_num"], now_json["range_num"]])
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        item = self.data[idx]

        source_filename = item[0]
        target_filename = item[1]
        heading_num = item[2]
        range_num = item[3]

        source = cv2.imread(source_filename)
        target = cv2.imread(target_filename)

        # Resize images to TARGET_SIZE x TARGET_SIZE first
        source = cv2.resize(source, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        target = cv2.resize(target, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Pad to PROCESS_SIZE for UNet compatibility (centered)
        source = pad_to_size(source, PROCESS_SIZE, pad_value=0)
        target = pad_to_size(target, PROCESS_SIZE, pad_value=0)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, heading_num=heading_num, range_num=range_num, hint=source)

