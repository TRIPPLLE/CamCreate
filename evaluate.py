import os
import tensorflow as tf
from models.custom_gan import CustomGenerator
from utils import get_edges2shoes_dataset
import gzip
from tqdm import tqdm

def evaluate_models(data_dir="dataset_stage1", custom_ckpt=None):
    print(f"Evaluating with TensorFlow {tf.__version__}")
    
    dataset = get_edges2shoes_dataset(data_dir, mode="val", batch_size=8)
    if dataset is None:
        print("Validation dataset not found, cannot evaluate.")
        return

    custom_gen = CustomGenerator(in_channels=3, out_channels=3)
    
    if custom_ckpt and os.path.exists(custom_ckpt):
        try:
            # Note: This requires a checkpoint manager or direct variable load
            # which depends on how it was saved.
            checkpoint = tf.train.Checkpoint(generator=custom_gen)
            checkpoint.restore(custom_ckpt).expect_partial()
        except Exception as e:
            print(f"Could not load checkpoint: {e}")
            
    cust_ssim, cust_psnr = 0.0, 0.0
    num_batches = 0
    
    print("Evaluating models...")
    for sketches, real_photos in tqdm(dataset):
        # The images are returned in [-1, 1], so we remap them to [0, 1] for TF image metrics
        real_photos_01 = (real_photos + 1.0) / 2.0
        
        # --- Custom ---
        # Generate and grab the highest resolution output (index 2)
        _, _, cust_fake = custom_gen(sketches, training=False)
        cust_fake_01 = (cust_fake + 1.0) / 2.0
        
        cust_ssim += tf.reduce_mean(tf.image.ssim(cust_fake_01, real_photos_01, max_val=1.0)).numpy()
        cust_psnr += tf.reduce_mean(tf.image.psnr(cust_fake_01, real_photos_01, max_val=1.0)).numpy()
        
        num_batches += 1
            
    if num_batches == 0:
        print("No validation batches found.")
        return
        
    # Compile Results
    results = {
        "Custom": {
            "SSIM": cust_ssim / num_batches,
            "PSNR": cust_psnr / num_batches
        }
    }
    
    print("\nEvaluation Results:")
    print("-" * 35)
    print(f"{'Metric':<10} | {'Custom (TensorFlow)':<20}")
    print("-" * 35)
    print(f"{'SSIM':<10} | {results['Custom']['SSIM']:<20.4f}")
    print(f"{'PSNR':<10} | {results['Custom']['PSNR']:<20.4f}")
    print("-" * 35)
    print("Note: FID and LPIPS computation require external libraries in TF (e.g. tensorflow-gan or custom VGG loops) and were omitted for simplicity.")

if __name__ == "__main__":
    evaluate_models()
