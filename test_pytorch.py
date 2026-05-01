import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from models.pytorch_gan import CustomGenerator
from pytorch_utils import get_dataloader, visualize_results

def test_pytorch(data_dir="dataset_stage1", checkpoint_path="checkpoints/pytorch/latest.pth.tar"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset & DataLoader
    # We use mode='val' to get validation images. 
    # The dataloader handles falling back to 'test' if 'val' doesn't exist.
    loader = get_dataloader(data_dir, mode='val', batch_size=4, num_workers=0)
    
    if loader.dataset is None or len(loader.dataset) == 0:
        print("Validation dataset not found, cannot test.")
        return

    # 2. Model
    generator = CustomGenerator().to(device)
    
    # 3. Load Checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['state_dict_G'])
    epoch = checkpoint.get('epoch', 'Unknown')
    print(f"Successfully loaded model from epoch {epoch}")

    # 4. Generate Images
    generator.eval()
    os.makedirs("outputs/pytorch/test", exist_ok=True)
    
    with torch.no_grad():
        for i, (sketches, real_photos) in enumerate(loader):
            sketches = sketches.to(device)
            real_photos = real_photos.to(device)
            
            # Generator outputs: out64, out128, fake_photos
            _, _, fake_photos = generator(sketches)
            
            save_path = f"outputs/pytorch/test/test_epoch_{epoch}_batch_{i}.png"
            visualize_results(sketches, real_photos, fake_photos, num_samples=4, save_path=save_path)
            print(f"Saved test image to {save_path}")
            
            # We just want to generate one test batch
            break

if __name__ == "__main__":
    test_pytorch()
