import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import gradio as gr
from models.pytorch_gan import CustomGenerator

# Initialize the model once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Loading PyTorch GAN on {device}...")

generator = CustomGenerator().to(device)

checkpoint_path = "checkpoints/pytorch/latest.pth.tar"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint['state_dict_G'])
    generator.eval()
    print(f"Successfully loaded checkpoint from epoch {checkpoint.get('epoch', 'Unknown')}")
else:
    print("Warning: No checkpoint found. Using untrained weights.")

# Define standard transformation (must match training pipeline)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def process_sketch(upload_img, sketch_dict):
    print("Generate button clicked!")
    # If the user pasted/uploaded an image, use that first
    if upload_img is not None:
        print("Using uploaded/pasted image")
        img = upload_img
    elif sketch_dict is not None:
        print("Using sketchpad image")
        # Gradio 4 ImageEditor returns a dict
        if isinstance(sketch_dict, dict) and "composite" in sketch_dict:
            img = sketch_dict["composite"]
        else:
            img = sketch_dict
    else:
        print("No image provided")
        return None
        
    # Ensure PIL Image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
        
    img = img.convert('RGB')
    
    # Threshold the image to pure black and white (removes gray pixels from screenshot anti-aliasing)
    # This ensures the input perfectly matches the training data distribution.
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    img_np = np.where(img_np < 200, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np).convert('RGB')
    
    # Pad to square with white background to maintain aspect ratio and precision
    width, height = img.size
    if width != height:
        max_dim = max(width, height)
        padded_img = Image.new('RGB', (max_dim, max_dim), 'white')
        padded_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        img = padded_img
    
    # Preprocess
    sketch_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Model returns out64, out128, fake_photos
        _, _, fake_photos = generator(sketch_tensor)
        
    # Post-process (Denormalize and convert to numpy image)
    generated = (fake_photos[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    
    return generated

# A blank white image to start the canvas
blank_canvas = Image.new("RGB", (256, 256), "white")

with gr.Blocks(title="SketchFlowGAN Interactive Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎨 SketchFlowGAN Drawing Interface")
    gr.Markdown("Provide a sketch on the left (either paste it or draw it), then click **Generate** to see the AI create a photorealistic version on the right!")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📥 Option 1: Paste or Upload Screenshot")
            upload_box = gr.Image(type="pil", label="Click here and press Ctrl+V to paste", sources=["upload", "clipboard"], interactive=True)
            
            gr.Markdown("### 🖍️ Option 2: Draw from scratch")
            sketchpad = gr.ImageEditor(
                value=blank_canvas,
                type="pil",
                image_mode="RGB",
                brush=gr.Brush(colors=["#000000"], default_size=3), # Black brush
                label="Draw your sketch here",
                interactive=True
            )
            generate_btn = gr.Button("Generate Realistic Photo", variant="primary")
            
        with gr.Column():
            output_image = gr.Image(label="Generated Photo", type="numpy", interactive=False)
            
    # Pass both the upload box and the sketchpad to the process function
    generate_btn.click(fn=process_sketch, inputs=[upload_box, sketchpad], outputs=output_image)

if __name__ == "__main__":
    print("Launching Gradio App. Go to http://127.0.0.0:7860 or http://localhost:7860 in your browser.")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
