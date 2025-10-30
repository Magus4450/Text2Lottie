import os
import argparse
from PIL import Image
import torch
from lavis.models import load_model_and_preprocess

def generate_caption(model, vis_processor, image: Image.Image, device: torch.device) -> str:
    image_tensor = vis_processor["eval"](image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.generate({"image": image_tensor})
    # output is a list of strings (depending on model). We pick first.
    caption = output[0]
    return caption

def process_folder(input_folder: str, output_folder: str,
                   model_name: str = "blip2_t5",
                   model_type: str = "caption_coco_flant5xl"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model & preprocessors
    model, vis_processors, _ = load_model_and_preprocess(
        name=model_name,
        model_type=model_type,
        is_eval=True,
        device=device
    )

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # supported image extensions
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

        caption = generate_caption(model, vis_processors, image, device)

        out_fn = base + ".txt"
        out_path = os.path.join(output_folder, out_fn)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(caption + "\n")

        print(f"  -> Saved caption to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch image captioning with BLIP2")
    parser.add_argument("input_folder", help="Folder containing input images")
    parser.add_argument("output_folder", help="Folder to save caption-text files")
    parser.add_argument("--model_name", default="blip2_t5", help="Name of BLIP2 model (LAVIS)")
    parser.add_argument("--model_type", default="caption_coco_flant5xl", help="Type of BLIP2 caption model")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder,
                   model_name=args.model_name, model_type=args.model_type)
