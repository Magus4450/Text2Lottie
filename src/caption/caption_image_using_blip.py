import os
import argparse
from PIL import Image
import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration

@torch.inference_mode()
def generate_caption(model, processor, image: Image.Image, device: torch.device) -> str:
    # Preprocess the image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    output_ids = model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(output_ids[0], skip_special_tokens=True)
    return caption


def process_folder(input_folder: str, output_folder: str,
                   model_name: str = "Salesforce/blip2-opt-2.7b-coco") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model: {model_name}")
    processor = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    model.eval()

    os.makedirs(output_folder, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for fn in os.listdir(input_folder):
        base, ext = os.path.splitext(fn)
        if ext.lower() not in exts:
            continue

        input_path = os.path.join(input_folder, fn)
        print(f"Captioning: {input_path}")
        try:
            image = Image.open(input_path).convert("RGB")
        except Exception as e:
            print(f"  Skipped {fn}: cannot open image ({e})")
            continue

        caption = generate_caption(model, processor, image, device)
        out_path = os.path.join(output_folder, base + ".txt")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(caption + "\n")

        print(f"  -> Saved caption to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image captioning with BLIP2 (HuggingFace Transformers)")
    parser.add_argument("input_folder", help="Folder containing input images")
    parser.add_argument("output_folder", help="Folder to save caption-text files")
    parser.add_argument("--model_name", default="Salesforce/blip2-opt-2.7b-coco",
                        help="BLIP2 model checkpoint on Hugging Face")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder, model_name=args.model_name)
