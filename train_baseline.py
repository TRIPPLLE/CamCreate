import os
import tensorflow as tf
from models.baseline_gan import EncoderDecoderGenerator, PatchGANDiscriminator, gan_loss_baseline
from utils import get_edges2shoes_dataset, visualize_batch, save_checkpoint, load_checkpoint

def train_baseline(data_dir="dataset_stage1", batch_size=16, epochs=100000, lr=0.0002):
    print(f"TensorFlow Version: {tf.__version__}")
    
    # Configure GPU memory growth
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Enabled memory growth for {len(gpus)} GPUs")
        except RuntimeError as e:
            print(e)
            
    # Enable mixed precision for faster training IF GPU is available
    if gpus:
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("Enabled mixed precision.")
    else:
        print("Running on CPU. Mixed precision disabled to avoid overflow warnings.")

    dataset = get_edges2shoes_dataset(data_dir, mode="train", batch_size=batch_size)
    if dataset is None:
        print(f"Dataset not found in '{data_dir}'! Exiting.")
        return

    # Init Models
    generator = EncoderDecoderGenerator(in_channels=3, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=6)
    
    # Optimizers
    opt_G = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    opt_D = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    
    lambda_L1 = 100.0

    # Ensure directories exist
    os.makedirs("checkpoints/baseline", exist_ok=True)
    os.makedirs("outputs/baseline", exist_ok=True)
    
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator, opt_G=opt_G, opt_D=opt_D)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, "checkpoints/baseline", max_to_keep=3)

    load_checkpoint(checkpoint, checkpoint_manager)

    @tf.function
    def train_step(sketches, real_photos):
        with tf.GradientTape(persistent=True) as tape:
            # Generator Output
            fake_photos = generator(sketches, training=True)
            
            # Discriminator predictions
            pred_real = discriminator(sketches, real_photos, training=True)
            pred_fake = discriminator(sketches, tf.stop_gradient(fake_photos), training=True)
            
            # Discriminator Loss
            loss_D_real = gan_loss_baseline(pred_real, is_real=True)
            loss_D_fake = gan_loss_baseline(pred_fake, is_real=False)
            
            loss_D = tf.cast((loss_D_real + loss_D_fake) * 0.5, tf.float32)
            
            # Generator predictions via Discriminator to compute loss
            pred_fake_for_G = discriminator(sketches, fake_photos, training=True)
            
            # Generator Loss
            loss_G_GAN = tf.cast(gan_loss_baseline(pred_fake_for_G, is_real=True), tf.float32)
            loss_G_L1 = tf.cast(tf.reduce_mean(tf.abs(fake_photos - real_photos)) * lambda_L1, tf.float32)
            
            loss_G = loss_G_GAN + loss_G_L1
            
        # Compute Gradients
        grads_D = tape.gradient(loss_D, discriminator.trainable_variables)
        grads_G = tape.gradient(loss_G, generator.trainable_variables)
        
        # Apply
        opt_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))
        opt_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        
        del tape # Drop the tape

        return loss_D, loss_G

    # Training Loop
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch}/{epochs}")
        for step, (sketches, real_photos) in enumerate(dataset):
            loss_D, loss_G = train_step(sketches, real_photos)
            
            if step % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {step}] [D loss: {loss_D:.4f}] [G loss: {loss_G:.4f}]")
                
                # Inference for visualization
                fake_photos = generator(sketches, training=False)
                visualize_batch(sketches, real_photos, fake_photos, num_samples=4, 
                                save_path=f"outputs/baseline/epoch_{epoch}_step_{step}.png")
                
        # Save checkpoint iteratively
        if epoch % 10 == 0:
            save_checkpoint(checkpoint_manager, epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_stage1", help="Path to dataset")
    args = parser.parse_args()
    train_baseline(data_dir=args.data_dir)
