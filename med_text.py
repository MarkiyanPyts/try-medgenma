from transformers import pipeline
import torch

pipe = pipeline(
    "text-generation",
    model="google/medgemma-27b-text-it",
    torch_dtype=torch.bfloat16,
    device="mps" if torch.backends.mps.is_available() else "cpu",
)

# Read test results from text.txt
with open("text.txt", "r") as f:
    test_results = f.read().strip()

messages = [
    {
        "role": "system",
        "content": "You are a helpful medical assistant providing insights on test results."
    },
    {
        "role": "user",
        "content": f"""Analyze these medical test results and provide comprehensive insights:

Test Results:
{test_results}

Please provide:
1. Key insights and interpretation of these test results
2. Any abnormal values or concerning findings
3. Which type of doctor or specialist the patient should visit
4. Reason for the specialist recommendation
5. Any immediate health concerns or urgent care needed
6. Recommended follow-up tests or examinations"""
    }
]

output = pipe(text=messages, max_new_tokens=500)
print("Medical Test Analysis")
print("=" * 50)
print(output[0]["generated_text"][-1]["content"])
