import os
from src import helper_functions
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from src.config import DOTA_CLASSES


class DotaDataset(Dataset):
    def __init__(self, root_dir, image_size=512, stride=4, augment=False, keep_difficult=False):
        self.root = root_dir
        self.image_size = image_size
        self.stride = stride
        self.augment = augment
        self.keep_difficult = keep_difficult
        self.class2id = {name: idx for idx, name in enumerate(DOTA_CLASSES)}
        self.num_classes = len(DOTA_CLASSES)
        self.images = self.find_all_images()
        self.labels = self.find_all_labels()
        self.stems = sorted(set(self.images) & set(self.labels))
        self.items = [(stem, self.images[stem], self.labels[stem]) for stem in self.stems]
        if not self.items:
            raise RuntimeError(f"No paired images and labels found in {self.root}")

    def __len__(self):
        return len(self.items)

    def find_all_images(self):
        results = {}
        images_path = os.path.join(self.root, "images")
        for img_name in os.listdir(images_path):
            if img_name.endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                stem = os.path.splitext(img_name)[0]
                img_path = os.path.join(images_path, img_name)
                results[stem] = img_path
        return results

    def find_all_labels(self):
        results = {}
        labels_path = os.path.join(self.root, "labels")
        for label_name in os.listdir(labels_path):
            if label_name.endswith(".txt"):
                stem = os.path.splitext(label_name)[0]
                label_path = os.path.join(labels_path, label_name)
                results[stem] = label_path
        return results

    def __getitem__(self, idx):
        stem, image_path, label_path = self.items[idx]
        image = Image.open(image_path)
        objects = helper_functions.parse_dota_label(label_path, self.class2id)
        if self.keep_difficult:
            for obj in objects:
                obj["difficult"] = 0
        image, objects = helper_functions.resize_image_and_objects(image, objects, self.image_size)
        if self.augment:
            image, objects = helper_functions.apply_train_augmentation(image, objects)

        image_array = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255.0
        image_tensor = torch.from_numpy(image_array)
        targets = helper_functions.make_targets(objects, self.image_size, self.stride, self.num_classes)

        meta = {"stem": stem, "image_path": str(image_path), "label_path": str(label_path), "objects": objects}
        return image_tensor, targets, meta
