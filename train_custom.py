import os
import tensorflow as tf
from models.custom_gan import CustomGenerator, CustomDiscriminator
from losses import CustomGANLosses
from utils import get_edges2shoes_dataset, visualize_batch, save_checkpoint, load_checkpoint

def gan_loss(out, is_real):
    if is_real:
        target = tf.ones_like(out)
    else:
        target = tf.zeros_like(out)
    return tf.reduce_mean(tf.square(out - target))

def train_custom(data_dir="dataset_stage1", batch_size=16, epochs=1000, lr=0.0002):
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

    num_samples = getattr(dataset, 'num_samples', float('inf'))
    if num_samples > 25000:
        steps_per_epoch = 25000 // batch_size
        dataset = dataset.take(steps_per_epoch)
        print("Limited dataset to 25000 samples for training.")
    else:
        steps_per_epoch = num_samples // batch_size
        
    print(f"Steps per epoch: {steps_per_epoch}")

    # Init Models
    generator = CustomGenerator(in_channels=3, out_channels=3)
    discriminator = CustomDiscriminator(in_channels=6)
    
    # Losses Manager
    losses_manager = CustomGANLosses()
    
    # Hyperparams for losses
    lambda_L1 = 100.0
    lambda_perceptual = 10.0
    lambda_fm = 10.0
    lambda_edge = 5.0

    # Optimizers
    opt_G = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    opt_D = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.5, beta_2=0.999)
    
    # Ensure directories exist
    os.makedirs("checkpoints/custom", exist_ok=True)
    os.makedirs("outputs/custom", exist_ok=True)
    
    checkpoint = tf.train.Checkpoint(generator=generator, discriminator=discriminator, opt_G=opt_G, opt_D=opt_D)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, "checkpoints/custom", max_to_keep=3)

    load_checkpoint(checkpoint, checkpoint_manager)

    @tf.function
    def train_step(sketches, real_photos):
        # Resize targets for multiscale Generator output
        real_128 = tf.image.resize(real_photos, [128, 128])
        real_64 = tf.image.resize(real_photos, [64, 64])

        with tf.GradientTape(persistent=True) as tape:
            # Generator Output
            out64, out128, fake_photos = generator(sketches, training=True)
            out64 = tf.cast(out64, tf.float32)
            out128 = tf.cast(out128, tf.float32)
            fake_photos = tf.cast(fake_photos, tf.float32)
            
            # Discriminator predictions
            out_global_real, out_local_real, feat_real = discriminator(sketches, real_photos, training=True)
            out_global_fake, out_local_fake, feat_fake = discriminator(sketches, tf.stop_gradient(fake_photos), training=True)
            
            # Discriminator Loss
            loss_D_real_g = gan_loss(out_global_real, is_real=True)
            loss_D_real_l = gan_loss(out_local_real, is_real=True)
            loss_D_fake_g = gan_loss(out_global_fake, is_real=False)
            loss_D_fake_l = gan_loss(out_local_fake, is_real=False)
            
            loss_D = tf.cast((loss_D_real_g + loss_D_real_l + loss_D_fake_g + loss_D_fake_l) * 0.5, tf.float32)
            
            # Generator predictions via Discriminator to compute loss
            out_global_fake_for_G, out_local_fake_for_G, feat_fake_for_G = discriminator(sketches, fake_photos, training=True)
            
            # 1. GAN Loss
            loss_G_GAN_g = gan_loss(out_global_fake_for_G, is_real=True)
            loss_G_GAN_l = gan_loss(out_local_fake_for_G, is_real=True)
            loss_G_GAN = tf.cast(loss_G_GAN_g + loss_G_GAN_l, tf.float32)
            
            # 2. Multi-Scale L1 Loss
            loss_G_L1 = tf.cast(tf.reduce_mean(tf.abs(out64 - real_64)) + \
                        tf.reduce_mean(tf.abs(out128 - real_128)) + \
                        tf.reduce_mean(tf.abs(fake_photos - real_photos)), tf.float32)
                        
            # 3. Perceptual Loss
            loss_G_VGG = tf.cast(losses_manager.perceptual(real_photos, fake_photos), tf.float32)
            
            # 4. Feature Matching Loss
            loss_G_FM = tf.cast(losses_manager.feature_matching_loss(feat_real, feat_fake_for_G), tf.float32)
            
            # 5. Edge Loss
            loss_G_Edge = tf.cast(losses_manager.edge_loss(real_photos, fake_photos), tf.float32)
            
            loss_G = (loss_G_GAN + 
                      loss_G_L1 * lambda_L1 + 
                      loss_G_VGG * lambda_perceptual + 
                      loss_G_FM * lambda_fm + 
                      loss_G_Edge * lambda_edge)
            
        # Compute Gradients
        grads_D = tape.gradient(loss_D, discriminator.trainable_variables)
        grads_G = tape.gradient(loss_G, generator.trainable_variables)
        
        # Apply
        opt_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))
        opt_G.apply_gradients(zip(grads_G, generator.trainable_variables))
        
        del tape # Drop the tape

        return loss_D, loss_G, loss_G_L1, loss_G_VGG, loss_G_FM

    # Training Loop
    for epoch in range(epochs):
        print(f"Starting Epoch {epoch}/{epochs}")
        for step, (sketches, real_photos) in enumerate(dataset):
            loss_D, loss_G, loss_L1, loss_VGG, loss_FM = train_step(sketches, real_photos)
            
            if step % 100 == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {step}/{steps_per_epoch}] "
                      f"[D: {loss_D:.4f} | G: {loss_G:.4f}] "
                      f"[L1: {loss_L1:.2f} VGG: {loss_VGG:.2f} FM: {loss_FM:.2f}]")
                
                # Inference for visualization
                _, _, fake_photos = generator(sketches, training=False)
                visualize_batch(sketches, real_photos, fake_photos, num_samples=4, 
                                save_path=f"outputs/custom/epoch_{epoch}_step_{step}.png")
                
        # Save checkpoint iteratively
        if epoch % 10 == 0:
            save_checkpoint(checkpoint_manager, epoch)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="dataset_stage1", help="Path to dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    args = parser.parse_args()
    train_custom(data_dir=args.data_dir, batch_size=args.batch_size)
