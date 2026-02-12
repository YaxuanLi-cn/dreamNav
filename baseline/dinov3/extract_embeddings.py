import os
import pickle
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel


def main():
    tours_dir = "/root/dreamNav/pairUAV/tours"
    embedding_dir = "./embedding"
    pretrained_model_name = "/root/dreamNav/models/dinov3_7b"

    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name,
        device_map="auto",
    )
    model.eval()

    image_extensions = {".jpeg", ".jpg", ".png", ".bmp", ".webp"}

    subfolders = sorted([
        d for d in os.listdir(tours_dir)
        if os.path.isdir(os.path.join(tours_dir, d))
    ])

    for subfolder in tqdm(subfolders, desc="Processing folders"):
        src_folder = os.path.join(tours_dir, subfolder)
        dst_folder = os.path.join(embedding_dir, subfolder)

        image_files = sorted([
            f for f in os.listdir(src_folder)
            if os.path.splitext(f)[1].lower() in image_extensions
        ])
        if not image_files:
            continue

        os.makedirs(dst_folder, exist_ok=True)

        for img_name in image_files:
            img_path = os.path.join(src_folder, img_name)
            stem = os.path.splitext(img_name)[0]
            pkl_path = os.path.join(dst_folder, stem + ".pkl")

            if os.path.exists(pkl_path):
                continue

            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, return_tensors="pt").to(model.device)

            with torch.inference_mode():
                outputs = model(**inputs)

            embedding = outputs.pooler_output.cpu()

            with open(pkl_path, "wb") as f:
                pickle.dump(embedding, f)


if __name__ == "__main__":
    main()
