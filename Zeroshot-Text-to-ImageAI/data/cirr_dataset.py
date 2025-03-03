import json
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

# Avoid PIL warning for large images
Image.MAX_IMAGE_PIXELS = None

class CIRRDataset(Dataset):
    """
    CIRRDataset handles data loading for the CIRR dataset in both 'relative' and 'classic' modes.
    - 'relative' mode provides tuples of (reference_image, target_image, caption) or names with additional metadata.
    - 'classic' mode provides image-name pairs.
    
    Attributes:
        split (str): Dataset split ['test1', 'train', 'val']
        mode (str): Mode for data ['relative', 'classic']
        preprocess (callable): Image preprocessing function (e.g., for normalization)
    """
    
    def __init__(self, split: str, mode: str, preprocess: callable, base_path: str = "/GPFS/data/yikunliu"):
        """
        Initialize the CIRR dataset with a given split, mode, and preprocess function.
        
        Args:
            split (str): Dataset split, must be one of ['test1', 'train', 'val']
            mode (str): Mode of operation, must be one of ['relative', 'classic']
            preprocess (callable): Preprocessing function for images
            base_path (str): Base directory where dataset files are located (default: '/GPFS/data/yikunliu')
        """
        self.cirr_path_prefix = base_path
        self.preprocess = preprocess
        self.mode = mode
        self.split = split if split != 'test_train' else 'train'  # Remap test_train to train internally
        
        self._validate_inputs()
        self._load_metadata()

        print(f"CIRR dataset initialized with split '{split}' in '{mode}' mode.")

    def _validate_inputs(self):
        """Validates the split and mode inputs to ensure correctness."""
        valid_splits = ['test1', 'train', 'val']
        valid_modes = ['relative', 'classic']
        
        if self.split not in valid_splits:
            raise ValueError(f"Invalid split '{self.split}', must be one of {valid_splits}")
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode '{self.mode}', must be one of {valid_modes}")
    
    def _load_metadata(self):
        """Loads the triplet and image-path metadata for the dataset based on the split."""
        try:
            triplet_file = os.path.join(self.cirr_path_prefix, f'CIRR/cirr/captions/cap.rc2.{self.split}.json')
            with open(triplet_file, 'r') as f:
                self.triplets = json.load(f)
            
            relpath_file = os.path.join(self.cirr_path_prefix, f'CIRR/cirr/image_splits/split.rc2.{self.split}.json')
            with open(relpath_file, 'r') as f:
                self.name_to_relpath = json.load(f)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required dataset files not found: {e}")
    
    def __len__(self):
        """Returns the length of the dataset depending on the mode."""
        return len(self.triplets) if self.mode == 'relative' else len(self.name_to_relpath)
    
    def __getitem__(self, index):
        """Retrieves an item from the dataset based on the mode and split."""
        try:
            if self.mode == 'relative':
                return self._get_relative_item(index)
            elif self.mode == 'classic':
                return self._get_classic_item(index)
            else:
                raise ValueError("Invalid mode, should be either 'relative' or 'classic'.")
        except Exception as e:
            print(f"Exception occurred at index {index}: {e}")
            return None

    def _get_relative_item(self, index):
        """Handles data retrieval in 'relative' mode based on the split."""
        data_point = self.triplets[index]
        group_members = data_point.get('img_set', {}).get('members', [])
        reference_name = data_point.get('reference')
        rel_caption = data_point.get('caption', '').lower()

        reference_image_path = self._get_image_path(reference_name)
        reference_image = self._load_image(reference_image_path)
        
        if self.split == 'train':
            target_name = data_point.get('target_hard')
            target_image_path = self._get_image_path(target_name)
            target_image = self._load_image(target_image_path)
            return reference_image, target_image, rel_caption
        elif self.split == 'val':
            target_name = data_point.get('target_hard')
            return reference_name, target_name, rel_caption, group_members, reference_image
        elif self.split == 'test1':
            pair_id = data_point.get('pairid')
            return pair_id, reference_name, rel_caption, group_members, reference_image
    
    def _get_classic_item(self, index):
        """Handles data retrieval in 'classic' mode."""
        image_name = list(self.name_to_relpath.keys())[index]
        image_path = self._get_image_path(image_name)
        image = self._load_image(image_path)
        return image_name, image
    
    def _get_image_path(self, image_name):
        """Returns the full image path based on the name."""
        if image_name not in self.name_to_relpath:
            raise ValueError(f"Image name '{image_name}' not found in image mappings.")
        return os.path.join(self.cirr_path_prefix, 'NLVR2/images/', self.name_to_relpath[image_name][2:])

    def _load_image(self, image_path):
        """Loads and preprocesses an image from the specified path."""
        try:
            with open(image_path, 'rb') as img_file:
                image = Image.open(img_file).convert('RGB')
                return self.preprocess(image)
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        except Exception as e:
            raise IOError(f"Failed to load image from {image_path}: {e}")
