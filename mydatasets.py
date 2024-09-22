import os
from PIL import Image
from torch.utils.data import Dataset


class CDIPDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Root directory containing the dataset (e.g., the directory with 'a', 'b', ..., 'z').
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._collect_image_paths()

    def _collect_image_paths(self):
        """
        Collect all .tif file paths from the dataset directory structure.
        """
        image_paths = []
        # Traverse through the alphabet directories ('a' to 'z')
        for alphabet_dir in os.listdir(self.root_dir):
            alphabet_path = os.path.join(self.root_dir, alphabet_dir)

            # Check if it's a directory
            if os.path.isdir(alphabet_path):
                # Traverse subdirectories inside each alphabet directory
                for sub_dir in os.listdir(alphabet_path):
                    sub_dir_path = os.path.join(alphabet_path, sub_dir)

                    # Check if it's a directory
                    if os.path.isdir(sub_dir_path):
                        # Look for .tif files in the subdirectory
                        for file_name in os.listdir(sub_dir_path):
                            if file_name.endswith('.tif'):
                                image_paths.append(os.path.join(sub_dir_path, file_name))

        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get an image by index.
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path)

        # Convert TIFF images to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply transformations if any
        if self.transform:
            image = self.transform(image)

        return image
