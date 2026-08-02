import os
import shutil
import random
from typing import List, Dict, Any, Tuple

class DatasetService:
    def split_dataset(
        self,
        source_dir: str = "dataset_manual",
        train_ratio: float = 0.7,
        val_ratio: float = 0.2,
        test_ratio: float = 0.1,
        base_dest: str = "dataset",
        class_names: List[str] = None
    ) -> Tuple[bool, str, Dict[str, Any]]:
        if class_names is None:
            class_names = ["car", "motorcycle", "bus", "truck"]

        images_dir = os.path.join(source_dir, "images")
        if not os.path.exists(images_dir):
            return False, f"Directory '{images_dir}' not found.", {}

        images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not images:
            return False, f"No images found in '{images_dir}' to split.", {}

        random.shuffle(images)

        # Normalize ratios if needed
        total_ratio = train_ratio + val_ratio + test_ratio
        if total_ratio > 0 and abs(total_ratio - 1.0) > 0.01:
            train_ratio /= total_ratio
            val_ratio /= total_ratio
            test_ratio /= total_ratio

        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)

        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:] if test_ratio > 0 else []

        # Create output dataset directory structure
        splits = ['train', 'val']
        if test_images:
            splits.append('test')

        for split in splits:
            os.makedirs(f"{base_dest}/{split}/images", exist_ok=True)
            os.makedirs(f"{base_dest}/{split}/labels", exist_ok=True)

        def copy_files(img_list, split_folder):
            for img in img_list:
                shutil.copy(os.path.join(source_dir, "images", img), os.path.join(base_dest, split_folder, "images"))
                label = img.rsplit('.', 1)[0] + '.txt'
                l_path = os.path.join(source_dir, "labels", label)
                if os.path.exists(l_path):
                    shutil.copy(l_path, os.path.join(base_dest, split_folder, "labels"))
                else:
                    with open(os.path.join(base_dest, split_folder, "labels", label), 'w') as f:
                        pass

        copy_files(train_images, 'train')
        copy_files(val_images, 'val')
        if test_images:
            copy_files(test_images, 'test')

        yaml_path = os.path.abspath(f"{base_dest}/traffic.yaml")
        names_yaml = "\n".join([f"  {i}: '{cls}'" for i, cls in enumerate(class_names)])
        
        test_yaml_line = "test: test/images\n" if test_images else ""

        yaml_content = f"""path: {os.path.abspath(base_dest)}
train: train/images
val: val/images
{test_yaml_line}
names:
{names_yaml}
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)

        summary = {
            "total_images": len(images),
            "train_count": len(train_images),
            "val_count": len(val_images),
            "test_count": len(test_images),
            "yaml_path": yaml_path,
            "classes": class_names
        }

        msg = f"Successfully created dataset split ({len(train_images)} train, {len(val_images)} val, {len(test_images)} test) and generated '{yaml_path}'"
        return True, msg, summary

dataset_service = DatasetService()
