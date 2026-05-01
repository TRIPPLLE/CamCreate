import os
import tensorflow as tf
import matplotlib.pyplot as plt

def get_edges2shoes_dataset(root_dir, mode="train", img_size=256, batch_size=16):
    """
    Returns a tf.data.Dataset for edges2shoes. The images in this dataset 
    are concatenated side-by-side [sketch, photo].
    """
    target_dir = os.path.join(root_dir, mode)
    if not os.path.exists(target_dir):
        if mode == "val":
            if os.path.exists(os.path.join(root_dir, "test")):
                target_dir = os.path.join(root_dir, "test")
            elif os.path.exists(os.path.join(root_dir, "val")):
                target_dir = os.path.join(root_dir, "val")
            else:
                return None
    
    if not os.path.exists(target_dir):
        return None

    # Load file paths
    import glob
    file_pattern = os.path.join(target_dir, "**", "*.jpg")
    file_paths = glob.glob(file_pattern, recursive=True)
    if not file_paths:
        file_paths = glob.glob(os.path.join(target_dir, "**", "*.png"), recursive=True)

    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if mode == "train":
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), 2000))

    def process_path(file_path):
        img = tf.io.read_file(file_path)
        # Decode image
        try:
            img = tf.image.decode_jpeg(img, channels=3)
        except:
            img = tf.image.decode_png(img, channels=3)
            
        shape = tf.shape(img)
        w = shape[1]
        
        # Split image into sketch (left) and photo (right)
        sketch = img[:, :w//2, :]
        photo = img[:, w//2:, :]

        # Resize
        sketch = tf.image.resize(sketch, [img_size, img_size])
        photo = tf.image.resize(photo, [img_size, img_size])

        # Normalize to [-1, 1]
        sketch = (sketch / 127.5) - 1.0
        photo = (photo / 127.5) - 1.0

        return sketch, photo

    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    # Store length as an attribute for reference based on number of batches
    dataset.num_samples = len(file_paths)
    return dataset

def visualize_batch(sketches, photos, generated=None, num_samples=4, save_path=None):
    """
    Visualizes a batch of original sketches, real photos, and optionally generated photos.
    """
    n = min(sketches.shape[0], num_samples)
    rows = 2 if generated is None else 3
    fig, axes = plt.subplots(rows, n, figsize=(n * 3, rows * 3))
    
    for i in range(n):
        # unnormalize and clip to [0, 1]
        sk = tf.clip_by_value((sketches[i] * 0.5 + 0.5), 0.0, 1.0).numpy()
        ph = tf.clip_by_value((photos[i] * 0.5 + 0.5), 0.0, 1.0).numpy()
        
        if n == 1:
            ax_sk = axes[0]
            ax_ph = axes[1]
            if generated is not None:
                ax_gen = axes[2]
        else:
            ax_sk = axes[0, i]
            ax_ph = axes[1, i]
            if generated is not None:
                ax_gen = axes[2, i]
                
        ax_sk.imshow(sk)
        ax_sk.axis('off')
        ax_sk.set_title("Sketch")
        
        ax_ph.imshow(ph)
        ax_ph.axis('off')
        ax_ph.set_title("Real Photo")
        
        if generated is not None:
            gn = tf.clip_by_value((generated[i] * 0.5 + 0.5), 0.0, 1.0).numpy()
            ax_gen.imshow(gn)
            ax_gen.axis('off')
            ax_gen.set_title("Generated")
            
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()
    return fig

def save_checkpoint(checkpoint_manager, epoch):
    """ Saves model via checkpoint manager """
    save_path = checkpoint_manager.save(checkpoint_number=epoch)
    print(f"Saved checkpoint for epoch {epoch}: {save_path}")

def load_checkpoint(checkpoint, manager):
    """ Loads model from checkpoint manager """
    checkpoint.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print(f"Restored from {manager.latest_checkpoint}")
    else:
        print("No checkpoint found. Initializing from scratch.")
