import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.optim as optim
import torch.nn.functional as F
from models.pytorch_gan import CustomGenerator, CustomDiscriminator
from pytorch_losses import PerceptualLoss, FeatureMatchingLoss, EdgeLoss, gan_loss
from pytorch_utils import get_sketchy_dataloader, visualize_results, save_checkpoint, load_checkpoint

def train_stage2(data_dir=r"dataset\temp_extraction\256x256", batch_size=32, epochs=100, lr=0.0001):
    # GPU Efficiency Tweaks
    torch.backends.cudnn.benchmark = True
    print("Enabled cuDNN Benchmark for optimal convolution algorithms.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 1. Dataset & DataLoader (Using higher workers and batch size for efficiency)
    loader = get_sketchy_dataloader(data_dir, mode='train', batch_size=batch_size, num_workers=8)
    steps_per_epoch = len(loader)
    print(f"Steps per epoch: {steps_per_epoch}")

    if steps_per_epoch == 0:
        print("Error: No data found. Please check data_dir.")
        return

    # 2. Models
    generator = CustomGenerator().to(device)
    discriminator = CustomDiscriminator().to(device)

    # 3. Optimizers (Lower learning rate for fine-tuning Stage 2)
    opt_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    # 4. Losses
    perceptual_fn = PerceptualLoss().to(device)
    fm_fn = FeatureMatchingLoss()
    edge_fn = EdgeLoss().to(device)
    
    scaler = torch.amp.GradScaler('cuda')
    
    lambda_L1 = 100.0
    lambda_perceptual = 10.0
    lambda_fm = 10.0
    lambda_edge = 5.0

    # 5. Directories
    os.makedirs("checkpoints/stage2", exist_ok=True)
    os.makedirs("outputs/stage2", exist_ok=True)
    
    stage2_ckpt = "checkpoints/stage2/latest.pth.tar"
    stage1_ckpt = "checkpoints/pytorch/latest.pth.tar"
    
    start_epoch = 0
    if os.path.exists(stage2_ckpt):
        print("Resuming Stage 2 from existing checkpoint...")
        start_epoch = load_checkpoint(stage2_ckpt, generator, discriminator, opt_G, opt_D)
    elif os.path.exists(stage1_ckpt):
        print("Starting Stage 2 using Stage 1 pretrained weights...")
        _ = load_checkpoint(stage1_ckpt, generator, discriminator, opt_G, opt_D)
        # Reset epoch to 0 for Stage 2
        start_epoch = 0
    else:
        print("WARNING: No Stage 1 weights found. Training from scratch.")

    # Skipping torch.compile as Triton is not fully supported on Windows.

    # 6. Training Loop
    try:
        for epoch in range(start_epoch, epochs):
            generator.train()
            discriminator.train()
            print(f"Starting Stage 2 - Epoch {epoch}/{epochs}")
            
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
                    print(f"[Stage 2] [Epoch {epoch}/{epochs}] [Batch {step}/{steps_per_epoch}] "
                          f"[D: {loss_D.item():.4f} | G: {loss_G.item():.4f}] "
                          f"[L1: {loss_G_L1.item():.2f} VGG: {loss_G_VGG.item():.2f}]")
                    
                    # Save preview
                    with torch.no_grad():
                        generator.eval()
                        # Use top 4 or less if batch is smaller
                        n_view = min(4, sketches.size(0))
                        _, _, preview_fake = generator(sketches[:n_view])
                        visualize_results(sketches[:n_view], real_photos[:n_view], preview_fake, 
                                          save_path=f"outputs/stage2/epoch_{epoch}_step_{step}.png")
                        generator.train()
    
            # Save checkpoint iteratively
            if epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_G': generator.state_dict(),
                    'state_dict_D': discriminator.state_dict(),
                    'optimizer_G': opt_G.state_dict(),
                    'optimizer_D': opt_D.state_dict(),
                }, filename=stage2_ckpt)
                
    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current checkpoint...")
        save_checkpoint({
            'epoch': epoch,
            'state_dict_G': generator.state_dict(),
            'state_dict_D': discriminator.state_dict(),
            'optimizer_G': opt_G.state_dict(),
            'optimizer_D': opt_D.state_dict(),
        }, filename=stage2_ckpt)
        print("Stage 2 Checkpoint saved successfully before exit.")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"dataset\temp_extraction\256x256", help="Path to Sketchy dataset")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Epoch limit")
    args = parser.parse_args()
    train_stage2(data_dir=args.data_dir, batch_size=args.batch_size, epochs=args.epochs)
