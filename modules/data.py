import torch
from torch.utils.data import Dataset
from torchvision import transforms
from einops import rearrange, repeat
from PIL import Image
import math
import random
import json

class Step1XEditDataset(Dataset):
    def __init__(
        self, 
        data_json = "/hy-tmp/training_1w_splited.json", 
        image_dir = "/hy-tmp/training_1w_splited", 
        image_size=512,
    ):
        """
        Args:
            data_json (list of dict): Each dict contains {"ref_image": ..., "tgt_image": ..., "prompt": ...}
            ref_image_dir (str): Path to reference images
            tgt_image_dir (str): Path to target images
            image_size (int): Input image size, default 512
        """
        if isinstance(data_json, list):
            self.data = []
            for data_json_item in data_json:
                self.data.extend(json.load(open(data_json_item, "r")))
        else:
            self.data = json.load(open(data_json, "r"))
        self.image_dir = image_dir
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.data)
    
    def input_process_image(self, img, img_size=512):
        # 1. 打开图片
        w, h = img.size
        r = w / h 

        if w > h:
            w_new = math.ceil(math.sqrt(img_size * img_size * r))
            h_new = math.ceil(w_new / r)
        else:
            h_new = math.ceil(math.sqrt(img_size * img_size / r))
            w_new = math.ceil(h_new * r)
        h_new = math.ceil(h_new) // 16 * 16
        w_new = math.ceil(w_new) // 16 * 16

        img_resized = img.resize((w_new, h_new))
        return img_resized, img.size

    def __getitem__(self, idx):
        entry = self.data[idx]
        ref_image_path = f"{self.image_dir}/{entry['folder_name']}/source.png"
        tgt_image_path = f"{self.image_dir}/{entry['folder_name']}/target.png"
        prompt = entry['prompt']

        # Load images
        ref_img_pil = Image.open(ref_image_path).convert('RGB')
        tgt_img_pil = Image.open(tgt_image_path).convert('RGB')
        ref_img_pil, ref_img_info = self.input_process_image(ref_img_pil)
        tgt_img_pil, tgt_img_info = self.input_process_image(tgt_img_pil)
        # Transform images
        ref_img = self.transform(ref_img_pil)  # (3, H, W)
        tgt_img = self.transform(tgt_img_pil)  # (3, H, W)

        return {
            "ref_img": ref_img.unsqueeze(0),
            "tgt_img": tgt_img.unsqueeze(0),
            "prompt": prompt,
            # "ref_img_pil": ref_img_pil,  # 给MLLM用原图输入
        }
