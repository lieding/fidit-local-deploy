from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
from PIL import Image
import torch
import argparse
weight_dtype = torch.bfloat16
device = "cuda"

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        required=True,
        help="image path",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        required=True,
        help="path to save",
    )
    args = parser.parse_args()

    return args


def process(image_path: str, save_path: str):
    
    image_encoder_large = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-large-patch14", torch_dtype=weight_dtype, cache_dir="models").to(device)
    image_encoder_bigG = CLIPVisionModelWithProjection.from_pretrained(
        "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k", torch_dtype=weight_dtype, cache_dir="models").to(device)
    vit_processing = CLIPImageProcessor()
    cloth_image = Image.open(image_path)
    cloth_image_vit = vit_processing(images=cloth_image, return_tensors="pt").data['pixel_values']
    cloth_image_vit = cloth_image_vit.to(device=device)
    image_embeds_large = image_encoder_large(cloth_image_vit).image_embeds
    image_embeds_bigG = image_encoder_bigG(cloth_image_vit).image_embeds
    embeds = torch.cat([image_embeds_large, image_embeds_bigG], dim=1)
    torch.save(embeds, save_path)

args = parse_args()
process(args.image_path, args.save_path)

