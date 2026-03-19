import torch
from PIL import Image
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

model_id = "google/paligemma2-3b-pt-448"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = PaliGemmaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

img1_path = "/hdd/sacson/Nov-17-2022-bww8_00000000_0/1.jpg"
img2_path = "/hdd/sacson/Nov-17-2022-bww8_00000000_0/30.jpg"
img1 = Image.open(img1_path).convert("RGB")
img2 = Image.open(img2_path).convert("RGB")

prompt = "how many seconds to travel from the first image to the second?\n"

inputs = processor(
    images=[[img1, img2]],
    text=prompt,
    return_tensors="pt",
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=20)

answer = processor.decode(
    output[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(answer)