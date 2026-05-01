import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import io
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from models.pytorch_gan import CustomGenerator
import uvicorn

app = FastAPI(title="SketchFlowGAN API")

# Add CORS so Flutter web/desktop can communicate without issues
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

@app.post("/generate")
async def generate_image(file: UploadFile = File(...)):
    print("Received generation request")
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # Threshold the image to pure black and white
    img_gray = img.convert('L')
    img_np = np.array(img_gray)
    img_np = np.where(img_np < 200, 0, 255).astype(np.uint8)
    img = Image.fromarray(img_np).convert('RGB')
    
    # Pad to square with white background
    width, height = img.size
    if width != height:
        max_dim = max(width, height)
        padded_img = Image.new('RGB', (max_dim, max_dim), 'white')
        padded_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
        img = padded_img
    
    # Preprocess
    sketch_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, fake_photos = generator(sketch_tensor)
        
    # Post-process
    generated = (fake_photos[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    generated_img = Image.fromarray((generated * 255).astype(np.uint8))
    
    # Convert back to bytes to send in HTTP response
    img_byte_arr = io.BytesIO()
    generated_img.save(img_byte_arr, format='JPEG', quality=95)
    img_byte_arr = img_byte_arr.getvalue()
    
    return Response(content=img_byte_arr, media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4567)
