# pip install accelerate
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import torch

model_id = "google/medgemma-4b-pt"

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Read test results from text.txt
with open("text.txt", "r") as f:
    test_results = f.read().strip()

# Create prompt asking for doctor recommendation
prompt = f"""Based on the following medical test results, please advise which type of doctor or specialist the patient should visit:

Test Results:
{test_results}

Please provide:
1. The type of specialist recommended
2. Reason for this recommendation
3. Any urgent concerns if present"""

inputs = processor(
    text=prompt, return_tensors="pt"
).to(model.device, dtype=torch.bfloat16)

input_len = inputs["input_ids"].shape[-1]

with torch.inference_mode():
    generation = model.generate(**inputs, max_new_tokens=200, do_sample=False)
    generation = generation[0][input_len:]

decoded = processor.decode(generation, skip_special_tokens=True)
print(decoded)
