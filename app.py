import torch
import clip
import cv2
import numpy as np
from PIL import Image, ImageDraw
from segment_anything import build_sam, SamAutomaticMaskGenerator
import gradio as gr

def segment(image_path: str, prompt: str, threshold: float = 0.05):
    mask_generator = SamAutomaticMaskGenerator(

    build_sam(checkpoint="sam_model_path/sam_vit_h_4b8939.pth"))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)

    image = Image.open(image_path)
    padding = 40
    cropped_boxes = []

    for mask in masks:
        img_array = np.array(image)
        segmented_array = np.zeros_like(img_array)
        segmented_array[mask["segmentation"]] = img_array[mask["segmentation"]]
        segmented_img = Image.fromarray(segmented_array)
        black_img = Image.new("RGB", image.size, (0, 0, 0))
        transparency_mask = np.zeros_like(mask["segmentation"], dtype=np.uint8)
        transparency_mask[mask["segmentation"]] = 255
        black_img.paste(segmented_img, mask=Image.fromarray(transparency_mask, mode='L'))
        box = [max(0, mask["bbox"][0] - padding),
               max(0, mask["bbox"][1] - padding),
               mask["bbox"][0] + mask["bbox"][2] + padding,
               mask["bbox"][1] + mask["bbox"][3] + padding]
        cropped_boxes.append(black_img.crop(box))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)

    preprocessed_images = [preprocess(img).to(device) for img in cropped_boxes]
    tokenized_text = clip.tokenize(prompt).to(device)
    stacked_images = torch.stack(preprocessed_images)
    img_features = model.encode_image(stacked_images)
    text_features = model.encode_text(tokenized_text)
    img_features /= img_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    probs =  100.*img_features @ text_features.T
    max_probs, max_indices = torch.max(probs, dim=1)
    selected_indices = [i for i, p in enumerate(max_probs) if p > threshold]
    selected_prompts = max_indices[selected_indices]
    best_prompts_and_scores = []

    for idx, prompt_idx in enumerate(selected_prompts):
        best_prompt_and_score = f"Contour {selected_indices[idx]}: Best prompt = '{prompt[prompt_idx]}', Score = {max_probs[selected_indices[idx]]:.4f}"
        best_prompts_and_scores.append(best_prompt_and_score)

    combined_mask = np.zeros_like(masks[0]["segmentation"], dtype=np.uint8)
    for seg_idx in selected_indices:
        combined_mask = np.logical_or(combined_mask, masks[seg_idx]["segmentation"])


    segmentation_masks = [Image.fromarray(masks[seg_idx]["segmentation"].astype('uint8') * 255) for seg_idx in
                          selected_indices]
    original_image = Image.open(image_path)
    overlay_image = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_image)
    for segmentation_mask_image in segmentation_masks:
        draw.bitmap((0, 0), segmentation_mask_image, fill=(255, 0, 0, 200))

        result_image= Image.alpha_composite(original_image.convert('RGBA'), overlay_image)

    return  result_image,best_prompts_and_scores


def gradio_segment(input_image: Image.Image, text_prompt: str, text_threshold: float):
    temp_image_path = "temp_input_image.png"
    input_image.save(temp_image_path)
    result = segment(temp_image_path, text_prompt.split(','), text_threshold)
    return result

def launch_gradio_interface():

    block = gr.Blocks().queue()
    with block:

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil")
                text_prompt = gr.Textbox(label="Prompts")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=False):
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )

            with gr.Column():
                gallery = gr.outputs.Image(
                    type="pil",
                ).style(full_width=True, full_height=True)
                result = gr.outputs.Textbox()

        run_button.click(fn=gradio_segment, inputs=[
            input_image, text_prompt, text_threshold], outputs=[gallery,result])

    block.launch()
if __name__ == '__main__':
    launch_gradio_interface()
