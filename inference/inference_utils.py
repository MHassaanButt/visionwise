import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# Model loading
CHECKPOINT = "microsoft/Florence-2-large-ft"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
model = AutoModelForCausalLM.from_pretrained(CHECKPOINT, trust_remote_code=True).to(
    DEVICE
)
processor = AutoProcessor.from_pretrained(CHECKPOINT, trust_remote_code=True)


# Inference function for image captioning
def run_inference(image: Image, task: str, text: str = ""):
    prompt = task + text
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE)
    print(f"Inputs Prepared for Model:\nText Input: {prompt}\nImage Size: {image.size}")
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task, image_size=image.size
    )
