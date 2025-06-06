from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load MedGemma text model (2B for text-only tasks)
model_id = "google/medgemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Read test results from file
with open("text.txt", "r") as f:
    test_results = f.read().strip()

# Create prompt asking for doctor recommendation
prompt = f"""Based on the following medical test results, please advise which type of doctor or specialist the patient should visit:

Test Results:
{test_results}

Please provide:
1. The type of specialist recommended
2. Reason for this recommendation
3. Any urgent concerns if present

Response:"""

# Tokenize and generate response
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.95
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
