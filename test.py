import base64
import math
import random
import os
from typing import List
from io import BytesIO

import torch
from PIL import Image
import numpy as np
#from huggingface_hub import snapshot_download

from preprocess.dwpose import DWposeDetector
from src.pose_guider import PoseGuider

from src.pipeline_stable_diffusion_3_tryon import StableDiffusion3TryOnPipeline
from src.transformer_sd3_garm import SD3Transformer2DModel as SD3Transformer2DModel_Garm
from src.transformer_sd3_vton import SD3Transformer2DModel as SD3Transformer2DModel_Vton

from test_cloth_embedding import default_cloth_embedding

fitdit_repo = "BoyuanJiang/FitDiT"
repo_path = snapshot_download(repo_id=fitdit_repo)

weight_dtype = torch.float16
device = "cuda"

pose_guider = PoseGuider(conditioning_embedding_channels=1536, conditioning_channels=3, block_out_channels=(32, 64, 256, 512))
pose_guider.load_state_dict(torch.load(os.path.join(repo_path, "pose_guider", "diffusion_pytorch_model.bin")))
pose_guider.to(device=device, dtype=weight_dtype)

dwprocessor = DWposeDetector(model_root=repo_path, device=device)

transformer_garm = SD3Transformer2DModel_Garm.from_pretrained(os.path.join(repo_path, "transformer_garm"), torch_dtype=weight_dtype)
transformer_vton = SD3Transformer2DModel_Vton.from_pretrained(os.path.join(repo_path, "transformer_vton"), torch_dtype=weight_dtype)
pipeline = StableDiffusion3TryOnPipeline.from_pretrained(
    repo_path,
    torch_dtype=weight_dtype,
    transformer_garm=transformer_garm,
    transformer_vton=transformer_vton,
    pose_guider=pose_guider)
pipeline.to(device)

def get_pose_img (vton_img: Image):
    vton_img_det = resize_image(vton_img)
    pose_image, keypoints, _, candidate = dwprocessor(np.array(vton_img_det)[:,:,::-1])
    candidate[candidate<0]=0
    candidate = candidate[0]

    candidate[:, 0]*=vton_img_det.width
    candidate[:, 1]*=vton_img_det.height

    pose_image = pose_image[:,:,::-1] #rgb
    pose_image = Image.fromarray(pose_image)

    return pose_image

def process(vton_img: Image, garm_img: Image, image_embeds_large, image_embeds_bigG, pre_mask_array, n_steps, image_scale, seed, num_images_per_prompt, resolution):
    assert resolution in ["768x1024", "1152x1536", "1536x2048"]
    new_width, new_height = resolution.split("x")
    new_width = int(new_width)
    new_height = int(new_height)
    with torch.inference_mode():

        pose_image = get_pose_img(vton_img)

        model_image_size = vton_img.size
        garm_img, _, _ = pad_and_resize(garm_img, new_width=new_width, new_height=new_height)
        vton_img, pad_w, pad_h = pad_and_resize(vton_img, new_width=new_width, new_height=new_height)

        mask = Image.fromarray(pre_mask_array)
        mask, _, _ = pad_and_resize(mask, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
        mask = mask.convert("L")
        #pose_image = Image.fromarray(pose_image)
        pose_image, _, _ = pad_and_resize(pose_image, new_width=new_width, new_height=new_height, pad_color=(0,0,0))
        if seed==-1:
            seed = random.randint(0, 2147483647)
        res = pipeline(
            height=new_height,
            width=new_width,
            guidance_scale=image_scale,
            num_inference_steps=n_steps,
            generator=torch.Generator("cpu").manual_seed(seed),
            cloth_image=garm_img,
            model_image=vton_img,
            image_embeds_large=image_embeds_large,
            image_embeds_bigG=image_embeds_bigG,
            mask=mask,
            pose_image=pose_image,
            num_images_per_prompt=num_images_per_prompt
        ).images
        for idx in range(len(res)):
            res[idx] = unpad_and_resize(res[idx], pad_w, pad_h, model_image_size[0], model_image_size[1])
        return res
    
def pad_and_resize(im, new_width=768, new_height=1024, pad_color=(255, 255, 255), mode=Image.LANCZOS):
    old_width, old_height = im.size
    
    ratio_w = new_width / old_width
    ratio_h = new_height / old_height
    if ratio_w < ratio_h:
        new_size = (new_width, round(old_height * ratio_w))
    else:
        new_size = (round(old_width * ratio_h), new_height)
    
    im_resized = im.resize(new_size, mode)

    pad_w = math.ceil((new_width - im_resized.width) / 2)
    pad_h = math.ceil((new_height - im_resized.height) / 2)

    new_im = Image.new('RGB', (new_width, new_height), pad_color)
    
    new_im.paste(im_resized, (pad_w, pad_h))

    return new_im, pad_w, pad_h

def unpad_and_resize(padded_im, pad_w, pad_h, original_width, original_height):
    width, height = padded_im.size
    
    left = pad_w
    top = pad_h
    right = width - pad_w
    bottom = height - pad_h
    
    cropped_im = padded_im.crop((left, top, right, bottom))

    resized_im = cropped_im.resize((original_width, original_height), Image.LANCZOS)

    return resized_im

def resize_image(img, target_size=768):
    width, height = img.size
    
    if width < height:
        scale = target_size / width
    else:
        scale = target_size / height
    
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    
    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
    
    return resized_img

def load_image_from_bytes(image_bytes: str):
    """
    Load an image from bytes data (e.g., from file upload or network request)
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        PIL.Image
    """
    image_stream = BytesIO(image_bytes)
    return Image.open(image_stream)

def load_image_from_base64(base64_string):
    """
    Load an image from base64 string
    
    Args:
        base64_string (str): Base64 encoded image string
        
    Returns:
        PIL.Image
    """
    # Decode base64 string
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    img_bytes = base64.b64decode(base64_string)
    return load_image_from_bytes(img_bytes)

def create_mask_with_borders(image_width: int, image_height: int, rect_x: int, rect_y: int, rect_width: int, rect_height: int):
    """
    Create a mask with transition borders where:
    - Inside area = 255
    - Border transition: outer = 40, inner = 128
    - Outside area = 0
    
    Args:
        image_width (int): Width of the image
        image_height (int): Height of the image
        rect_x (int): X coordinate of top-left corner
        rect_y (int): Y coordinate of top-left corner
        rect_width (int): Width of rectangle
        rect_height (int): Height of rectangle
    
    Returns:
        numpy.ndarray: Mask array with transition borders
    """
    # Create base array filled with zeros
    mask = np.zeros((image_height, image_width), dtype=np.uint8)
    
    # Ensure coordinates are within bounds
    x1 = max(0, rect_x)
    y1 = max(0, rect_y)
    x2 = min(image_width, rect_x + rect_width)
    y2 = min(image_height, rect_y + rect_height)
    
    # Fill inner rectangle with 255
    mask[y1+2:y2-2, x1+2:x2-2] = 255
    
    # Create inner border (128)
    if y1+2 < y2-2 and x1+2 < x2-2:
        # Top and bottom inner borders
        mask[y1+1:y1+2, x1+1:x2-1] = 128
        mask[y2-2:y2-1, x1+1:x2-1] = 128
        # Left and right inner borders
        mask[y1+1:y2-1, x1+1:x1+2] = 128
        mask[y1+1:y2-1, x2-2:x2-1] = 128
    
    # Create outer border (40)
    if y1 < y2 and x1 < x2:
        # Top and bottom outer borders
        mask[y1:y1+1, x1:x2] = 40
        mask[y2-1:y2, x1:x2] = 40
        # Left and right outer borders
        mask[y1:y2, x1:x1+1] = 40
        mask[y1:y2, x2-1:x2] = 40
    
    return mask

def for_api_call(
    img_width: int,
    img_height: int,
    rect_x: int,
    rect_y: int,
    rect_width: int,
    rect_height: int,
    vton_img_base64: str,
    cloth_img_base64: str,
    image_embeds_large: List,
    image_embeds_bigG: List,
    step_nums: int = 20,
    guidance: int = 2,
    batch: int = 1,
    resolution_str: str = "768x1024"
)-> Image:
    """
    Do vittual try-on for api calling
    
    Args:
        img_width (int): Width of the image
        ing_height (int): Height of the image
        rect_x (int): X coordinate of top-left corner
        rect_y (int): Y coordinate of top-left corner
        rect_width (int): Width of rectangle
        rect_height (int): Height of rectangle
        vton_img_base64 (str): The model image in base64 format
        cloth_img_base64 (str): thr cloth image in base64 format
    
    Returns:
        PIL.Image
    """
    mask_array = create_mask_with_borders(img_width, img_height, rect_x, rect_y, rect_width, rect_height)
    vton_img_base64 = load_image_from_base64(vton_img_base64)
    cloth_img_base64 = load_image_from_base64(cloth_img_base64)
    imgs = process(vton_img_base64, cloth_img_base64, image_embeds_large, image_embeds_bigG, mask_array, step_nums, guidance, -1, batch, resolution_str)
    img = imgs[0]
    image_stream = BytesIO()
    img.save(image_stream, format="WebP")  # Change format if needed (JPEG, PNG, etc.)

    # Step 2: Encode the binary data to Base64
    base64_bytes = base64.b64encode(image_stream.getvalue())

    # Step 3: Convert to a UTF-8 string
    return base64_bytes.decode("utf-8")


# if __name__ == "__main__":
#     category = "Upper-body" # "Upper-body", "Lower-body", "Dresses"
#     array = create_mask_with_borders(960, 1280, 263, 271, 465, 393)
#     image_embeds_large=default_cloth_embedding['large']
#     image_embeds_bigG=default_cloth_embedding['bigG']
#     imgs = process(Image.open('../vton.png'), Image.open('../cloth.jpg'), image_embeds_large, image_embeds_bigG, array, 20, 2, -1, 1, "768x1024")
#     for i in range(len(imgs)):
#         imgs[i].save('../' + str(i) + '.jpg')
