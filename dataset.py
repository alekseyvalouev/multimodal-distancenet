# Need a dataset class for our language classification task. 

import torch
from torch.utils.data import Dataset

import os
import json
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt

class SacsonDataset(Dataset):
    def __init__(self, scenes, horizons=[2, 4, 8, 16], negative_samples=True, transform=None, classification=True):
        # We assume the mean distance between samples is 0.2 meters. 
        # This means that the label is delta(indices) * 0.4
        self.scale_factor = 0.4
        self.horizons = []
        self.labels = {}
        for i, h in enumerate(horizons):
            assert h / self.scale_factor % 1 == 0, "Horizon must be a multiple of 0.4 (scaling factor for Sacson)"
            self.horizons.append(int(h / self.scale_factor))
            self.labels[h] = i
        self.scenes = scenes
        self.negative_samples = negative_samples
        self.transform = transform
        self.classification = classification
        self._load_data()
        if self.negative_samples:
            self._load_negative_samples()
            self.labels[-1] = len(self.horizons)
    
    def _load_data(self):
        self.data = []
        self.data_path = "/hdd/sacson"
        for scene in self.scenes:
            folder = os.path.join(self.data_path, scene)
            # we're in the folder. get the highest #'ed image. 
            images = list(filter(lambda x: x.endswith(".jpg"), os.listdir(folder)))
            images.sort(key=lambda x: int(x.split(".")[0]))
            highest_image = images[-1]
            max_idx = int(highest_image.split(".")[0])

            for h in self.horizons:
                for i in range(max_idx - h + 1):
                    start_idx = i
                    end_idx = i + h
                    start_img = images[start_idx]
                    end_img = images[end_idx]
                    start_img = os.path.join(folder, start_img)
                    end_img = os.path.join(folder, end_img)
                    self.data.append((start_img, end_img, h))

        print(f"Loaded dataset with {len(self.data)} samples.")
    
    def _make_random_sample(self, scene):
        folder = os.path.join(self.data_path, scene)
        images = list(filter(lambda x: x.endswith(".jpg"), os.listdir(folder)))
        image = random.choice(images)
        return os.path.join(folder, image)

    def _load_negative_samples(self):
        self.negative_samples = []
        p = 0.25
        for _ in range(int(len(self.data) * p)):
            scene_start, scene_end = np.random.choice(self.scenes, size=2, replace=False)
            start_img = self._make_random_sample(scene_start)
            end_img = self._make_random_sample(scene_end)
            self.negative_samples.append((start_img, end_img, -1))
            
        self.data.extend(self.negative_samples)

        print(f"Generated {len(self.negative_samples)} negative samples.")
    
    def _get_image(self, image_path):
        img = Image.open(image_path).convert("RGB")

        # Resize so that the smallest dimension is 224 while preserving aspect ratio
        width, height = img.size
        scale = 224.0 / min(width, height)
        new_width = int(round(width * scale))
        new_height = int(round(height * scale))
        img = img.resize((new_width, new_height), Image.BILINEAR)

        # Center crop to 224x224
        left = (new_width - 224) // 2
        top = (new_height - 224) // 2
        right = left + 224
        bottom = top + 224
        img = img.crop((left, top, right, bottom))

        return np.array(img)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        start_img, end_img, horizon = self.data[idx]
        start_name, end_name = start_img.split("/")[-1], end_img.split("/")[-1]
        start_scene, end_scene = start_img.split("/")[-2], end_img.split("/")[-2]
        start_img = self._get_image(start_img)
        end_img = self._get_image(end_img)
        if self.transform:
            start_img = self.transform(start_img)
            end_img = self.transform(end_img)
        label = horizon
        if self.classification:
            label = self.labels[horizon]
        if label == -1:
            label = max(self.horizons)
        return start_name, end_name, start_scene, end_scene, start_img, end_img, int(label*self.scale_factor)


class LanguageDistanceDataset(Dataset):
    def __init__(self, annotations_folder, horizons=[2, 4, 8, 16], transform=None, lang_labels=True):
        self.annotations_folder = annotations_folder
        self.horizons = horizons
        self.transform = transform
        self.lang_labels = lang_labels
        self._load_data()

        # now load the actual sacson loader
        self.sacson_data = SacsonDataset(self.scenes, self.horizons, transform=self.transform, classification=False)
    
    def _load_data(self):
        # load self.scenes
        self.scenes = []
        # load self.annotations 
        self.annotations = {}
        # iterate through the annotations folder and load each json
        for annotation_file in os.listdir(self.annotations_folder):
            scene_name = annotation_file.replace("_landmarks.json", "")
            self.scenes.append(scene_name)
            with open(os.path.join(self.annotations_folder, annotation_file), "r") as f:
                annotation = json.load(f)
            self.annotations[scene_name] = annotation["landmarks"]
    
    def _stitch_images(self, img1, img2):
        stitched_img = np.concatenate([img1, img2], axis=1)
        return stitched_img

    def __len__(self):
        return len(self.sacson_data)
    
    def __getitem__(self, idx):
        # I think transforms will actually just break this :(
        start_name, end_name, start_scene, end_scene, start_img, end_img, label = self.sacson_data[idx]
        input_image = self._stitch_images(start_img, end_img)
        start_landmarks = self.annotations[start_scene][start_name]
        end_landmarks = self.annotations[end_scene][end_name]
        start_landmarks_str = " ".join([f"{i+1}. {landmark}" for i, landmark in enumerate(start_landmarks)])
        end_landmarks_str = " ".join([f"{i+1}. {landmark}" for i, landmark in enumerate(end_landmarks)])
        if self.lang_labels:
            return {
            "image": input_image,
                "prefix": f"answer en Starting image: {start_landmarks_str} Ending image: {end_landmarks_str}. What is the temporal distance?\n",
                "suffix": f"{label}"
            }
        else:
            return {
                "image": input_image,
                "prefix": f"answer en What is the temporal distance?\n",
                "suffix": f"{label}"
            }

if __name__ == "__main__":
    dataset = LanguageDistanceDataset(annotations_folder="/home/alekseyvalouev/goalnav/language-annotations-train")
    sample = dataset[0]

    image = sample["image"]
    prompt = sample["prefix"]
    label = sample["suffix"]

    plt.figure(figsize=(8, 8))
    plt.imshow(image.astype(np.uint8))
    plt.axis("off")
    plt.title(f"Label: {label}")

    # Put the (possibly long) prompt as a caption under the image.
    plt.figtext(
        0.5,
        0.02,
        prompt,
        wrap=True,
        horizontalalignment="center",
        fontsize=8,
    )

    plt.tight_layout()
    #plt.show()
    plt.savefig("language_distance_sample.png")