from transformers import pipeline
from PIL import Image
import requests
import torch

pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-4b-pt",
    torch_dtype=torch.bfloat16,
    device="mps" if torch.backends.mps.is_available() else "cpu",
)

# Load local X-ray image
image_path = "IMG_3380.jpg"  # Your X-ray image
image = Image.open(image_path)

output = pipe(
    images=image,
    text="<start_of_image> findings:",
    max_new_tokens=100,
)
print(output[0]["generated_text"])
