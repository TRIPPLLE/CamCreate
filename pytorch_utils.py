import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import glob

class Edges2ShoesDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=256):
        self.target_dir = os.path.join(root_dir, mode)
        if not os.path.exists(self.target_dir):
            if mode == 'val':
                if os.path.exists(os.path.join(root_dir, 'test')):
                    self.target_dir = os.path.join(root_dir, 'test')
                elif os.path.exists(os.path.join(root_dir, 'val')):
                    self.target_dir = os.path.join(root_dir, 'val')
        
        self.file_paths = glob.glob(os.path.join(self.target_dir, "**", "*.jpg"), recursive=True)
        if not self.file_paths:
            self.file_paths = glob.glob(os.path.join(self.target_dir, "**", "*.png"), recursive=True)
            
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        w, h = img.size
        # Split side-by-side [sketch, photo]
        sketch_img = img.crop((0, 0, w // 2, h))
        photo_img = img.crop((w // 2, 0, w, h))
        
        sketch = self.transform(sketch_img)
        photo = self.transform(photo_img)
        
        return sketch, photo

def get_dataloader(root_dir, mode='train', batch_size=16, img_size=256, num_workers=5):
    dataset = Edges2ShoesDataset(root_dir, mode, img_size)
    
    # Prefetching and persistent workers only for training
    loader_args = {
        'batch_size': batch_size,
        'shuffle': (mode == 'train'),
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True
    }
    
    if mode == 'train' and num_workers > 0:
        loader_args.update({
            'prefetch_factor': 2,
            'persistent_workers': True
        })
        
    return DataLoader(dataset, **loader_args)

class SketchyDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=256):
        # root_dir is expected to be d:\PORJECT\CAMCReate\dataset\temp_extraction\256x256
        self.sketches_dir = os.path.join(root_dir, 'splitted_sketches', mode, 'tx_000000000000')
        self.photos_dir = os.path.join(root_dir, 'photo', 'tx_000000000000')
        
        # Fallback to test if val requested and doesn't exist
        if mode == 'val' and not os.path.exists(self.sketches_dir):
            self.sketches_dir = os.path.join(root_dir, 'splitted_sketches', 'test', 'tx_000000000000')
            
        self.sketch_paths = glob.glob(os.path.join(self.sketches_dir, "**", "*.png"), recursive=True)
        if not self.sketch_paths:
            self.sketch_paths = glob.glob(os.path.join(self.sketches_dir, "**", "*.jpg"), recursive=True)
            
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.sketch_paths)

    def __getitem__(self, idx):
        sketch_path = self.sketch_paths[idx]
        
        # Determine the corresponding photo path
        # Example sketch_path: .../airplane/n02691156_10151-1.png
        class_name = os.path.basename(os.path.dirname(sketch_path))
        sketch_filename = os.path.basename(sketch_path)
        
        # Extract base photo name (remove -x suffix)
        base_name = sketch_filename.rsplit('-', 1)[0]
        photo_path_jpg = os.path.join(self.photos_dir, class_name, base_name + ".jpg")
        photo_path_png = os.path.join(self.photos_dir, class_name, base_name + ".png")
        
        if os.path.exists(photo_path_jpg):
            photo_path = photo_path_jpg
        else:
            photo_path = photo_path_png
            
        sketch_img = Image.open(sketch_path).convert('RGB')
        
        try:
            photo_img = Image.open(photo_path).convert('RGB')
        except FileNotFoundError:
            # Fallback if somehow photo is missing, return a blank or skip
            # In a real scenario we'd pre-filter the dataset
            photo_img = Image.new('RGB', (256, 256), 'white')

        sketch = self.transform(sketch_img)
        photo = self.transform(photo_img)
        
        return sketch, photo

def get_sketchy_dataloader(root_dir, mode='train', batch_size=32, img_size=256, num_workers=8):
    dataset = SketchyDataset(root_dir, mode, img_size)
    
    loader_args = {
        'batch_size': batch_size,
        'shuffle': (mode == 'train'),
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True
    }
    
    if mode == 'train' and num_workers > 0:
        loader_args.update({
            'prefetch_factor': 2,
            'persistent_workers': True
        })
        
    return DataLoader(dataset, **loader_args)


def visualize_results(sketches, photos, generated=None, num_samples=4, save_path=None):
    n = min(sketches.size(0), num_samples)
    rows = 2 if generated is None else 3
    fig, axes = plt.subplots(rows, n, figsize=(n * 3, rows * 3))
    
    def denorm(x):
        return (x * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()

    for i in range(n):
        ax_sk = axes[0, i] if n > 1 else axes[0]
        ax_ph = axes[1, i] if n > 1 else axes[1]
        
        ax_sk.imshow(denorm(sketches[i]))
        ax_sk.axis('off')
        ax_sk.set_title("Sketch")
        
        ax_ph.imshow(denorm(photos[i]))
        ax_ph.axis('off')
        ax_ph.set_title("Real Photo")
        
        if generated is not None:
            ax_gen = axes[2, i] if n > 1 else axes[2]
            ax_gen.imshow(denorm(generated[i]))
            ax_gen.axis('off')
            ax_gen.set_title("Generated")
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(checkpoint_path, model_G, model_D, opt_G, opt_D):
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found. Starting from scratch.")
        return 0
    
    checkpoint = torch.load(checkpoint_path)
    model_G.load_state_dict(checkpoint['state_dict_G'])
    model_D.load_state_dict(checkpoint['state_dict_D'])
    opt_G.load_state_dict(checkpoint['optimizer_G'])
    opt_D.load_state_dict(checkpoint['optimizer_D'])
    return checkpoint['epoch']
