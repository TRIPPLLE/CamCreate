import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.optim as optim
from torch.utils.data import Subset
from models.pytorch_gan import CustomGenerator, CustomDiscriminator
from pytorch_losses import PerceptualLoss, FeatureMatchingLoss, EdgeLoss, gan_loss
from pytorch_utils import get_dataloader, visualize_results, save_checkpoint, load_checkpoint

def train_pytorch(data_dir="dataset_stage1", batch_size=16, epochs=1000, lr=0.0002):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & DataLoader (Using 5 workers for balanced RTX 5070 performance)
    full_loader = get_dataloader(data_dir, mode='train', batch_size=batch_size, num_workers=5)
    dataset = full_loader.dataset
    
    # Apply 25k limit for Stage 1 if needed
    if len(dataset) > 25000:
        print(f"Limiting dataset from {len(dataset)} to 25000 samples for Stage 1.")
        indices = list(range(25000))
        dataset = Subset(dataset, indices)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=5, 
            pin_memory=True, drop_last=True, prefetch_factor=2, persistent_workers=True
        )
    else:
        loader = full_loader

    steps_per_epoch = len(loader)
    print(f"Steps per epoch: {steps_per_epoch}")

    # 2. Models
    generator = CustomGenerator().to(device)
    discriminator = CustomDiscriminator().to(device)
    
    # 3. Optimizers
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 4. Losses
    perceptual_fn = PerceptualLoss().to(device)
    fm_fn = FeatureMatchingLoss()
    edge_fn = EdgeLoss().to(device)
    
    # 4.1 Scaler for Mixed Precision (Modern Syntax)
    scaler = torch.amp.GradScaler('cuda')
    
    lambda_L1 = 100.0
    lambda_perceptual = 10.0
    lambda_fm = 10.0
    lambda_edge = 5.0

    # 5. Directories
    os.makedirs("checkpoints/pytorch", exist_ok=True)
    os.makedirs("outputs/pytorch", exist_ok=True)
    
    checkpoint_path = "checkpoints/pytorch/latest.pth.tar"
    start_epoch = load_checkpoint(checkpoint_path, generator, discriminator, opt_G, opt_D)

    # 6. Training Loop
    try:
        for epoch in range(start_epoch, epochs):
            generator.train()
            discriminator.train()
            print(f"Starting Epoch {epoch}/{epochs}")
            
            for step, (sketches, real_photos) in enumerate(loader):
                sketches = sketches.to(device)
                real_photos = real_photos.to(device)
                
                # --- Train Discriminator ---
                opt_D.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # Multi-scale Resize for real targets
                    real_128 = F.interpolate(real_photos, size=(128, 128), mode='bilinear', align_corners=False)
                    real_64 = F.interpolate(real_photos, size=(64, 64), mode='bilinear', align_corners=False)
                    
                    # Generator outputs
                    out64, out128, fake_photos = generator(sketches)
                    
                    # Discriminator predictions
                    pred_global_real, pred_local_real, _ = discriminator(sketches, real_photos)
                    pred_global_fake, pred_local_fake, _ = discriminator(sketches, fake_photos.detach())
                    
                    loss_D_real = (gan_loss(pred_global_real, True) + gan_loss(pred_local_real, True)) * 0.5
                    loss_D_fake = (gan_loss(pred_global_fake, False) + gan_loss(pred_local_fake, False)) * 0.5
                    loss_D = loss_D_real + loss_D_fake
                
                scaler.scale(loss_D).backward()
                scaler.step(opt_D)
                # opt_D.step() scaler.step handles it
                
                # --- Train Generator ---
                opt_G.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    # Predictions via D for G loss
                    pred_global_fake_for_G, pred_local_fake_for_G, feat_fake = discriminator(sketches, fake_photos)
                    _, _, feat_real = discriminator(sketches, real_photos)
                    
                    # 1. GAN Loss
                    loss_G_GAN = (gan_loss(pred_global_fake_for_G, True) + gan_loss(pred_local_fake_for_G, True))
                    
                    # 2. Multi-Scale L1 Loss
                    loss_G_L1 = (F.l1_loss(out64, real_64) + 
                                 F.l1_loss(out128, real_128) + 
                                 F.l1_loss(fake_photos, real_photos))
                    
                    # 3. Perceptual Loss
                    loss_G_VGG = perceptual_fn(real_photos, fake_photos)
                    
                    # 4. Feature Matching
                    loss_G_FM = fm_fn(feat_real, feat_fake)
                    
                    # 5. Edge Loss
                    loss_G_Edge = edge_fn(real_photos, fake_photos)
                    
                    loss_G = (loss_G_GAN + 
                              loss_G_L1 * lambda_L1 + 
                              loss_G_VGG * lambda_perceptual + 
                              loss_G_FM * lambda_fm + 
                              loss_G_Edge * lambda_edge)
                
                scaler.scale(loss_G).backward()
                scaler.step(opt_G)
                scaler.update()
                
                if step % 50 == 0:
                    print(f"[Epoch {epoch}/{epochs}] [Batch {step}/{steps_per_epoch}] "
                          f"[D: {loss_D.item():.4f} | G: {loss_G.item():.4f}] "
                          f"[L1: {loss_G_L1.item():.2f} VGG: {loss_G_VGG.item():.2f}]")
                    
                    # Save preview
                    with torch.no_grad():
                        generator.eval()
                        _, _, preview_fake = generator(sketches[:4])
                        visualize_results(sketches[:4], real_photos[:4], preview_fake, 
                                          save_path=f"outputs/pytorch/epoch_{epoch}_step_{step}.png")
                        generator.train()
    
            # Save checkpoint iteratively
            if epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_G': generator.state_dict(),
                    'state_dict_D': discriminator.state_dict(),
                    'optimizer_G': opt_G.state_dict(),
                    'optimizer_D': opt_D.state_dict(),
                }, filename=checkpoint_path)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current checkpoint...")
        save_checkpoint({
            'epoch': epoch,
            'state_dict_G': generator.state_dict(),
            'state_dict_D': discriminator.state_dict(),
            'optimizer_G': opt_G.state_dict(),
            'optimizer_D': opt_D.state_dict(),
        }, filename=checkpoint_path)
        print("Checkpoint saved successfully before exit.")
        raise

if __name__ == "__main__":
    import argparse
    import torch.nn.functional as F
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_stage1", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    train_pytorch(data_dir=args.data_dir, batch_size=args.batch_size)
