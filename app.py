import gradio as gr
import torch
import diffusers
from utils import patch_attention_proc
import math
import numpy as np
from PIL import Image

pipe = diffusers.StableDiffusionPipeline.from_pretrained("Lykon/DreamShaper").to("cuda", torch.float16)
pipe.enable_xformers_memory_efficient_attention()
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None


with gr.Blocks() as demo:
    prompt = gr.Textbox(interactive=True, label="prompt")
    negative_prompt = gr.Textbox(interactive=True, label="negative_prompt")
    guidance_scale = gr.Number(label="guidance_scale", value=7.5, precision=1)
    height = gr.Number(label="height", value=1024, precision=0)
    width = gr.Number(label="width", value=1024, precision=0)
    steps = gr.Number(label="steps", value=20, precision=0)
    seed = gr.Number(label="seed", value=1, precision=0)

    output_image = gr.Image(label=f"output_image", type="pil", interactive=False)

    gen = gr.Button("generate")


    def which_image(img, target_val=253, width=1024):
        npimg = np.array(img)
        loc = np.where(npimg[:, :, 3] == target_val)[1].item()
        if loc > width:
            print("Right Image is merged!")
        else:
            print("Left Image is merged!")


    def generate(prompt, seed, steps, height, width, negative_prompt, guidance_scale):

        pipe.enable_xformers_memory_efficient_attention()

        downsample_factor = 2
        ratio = 0.38
        merge_method = "downsample"

        if math.sqrt(height * width) > 1024:
            downsample_factor = 3
            ratio = 0.5
            downsample_factor_level_2 = 1
            ratio_level_2 = 0.0
        if math.sqrt(height * width) > 1400:
            downsample_factor = 4
            ratio = 0.6
            downsample_factor_level_2 = 2
            ratio_level_2 = 0.5
        if math.sqrt(height * width) < 804:
            merge_method = "similarity"
            ratio = 0.38
            downsample_factor_level_2 = 1
            ratio_level_2 = 0.0

        token_merge_args = {"ratio": ratio,
                            "merge_tokens": "keys/values",
                            "merge_method": merge_method,
                            "downsample_method": "nearest-exact",
                            "downsample_factor": downsample_factor,
                            "timestep_threshold_switch": 0.3,
                            "timestep_threshold_stop": 0.0,
                            "downsample_factor_level_2": downsample_factor_level_2,
                            "ratio_level_2": ratio_level_2
                            }

        l_r = torch.rand(1).item()

        torch.manual_seed(seed)

        base_img = pipe(prompt,
                        num_inference_steps=steps, height=height, width=width,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale).images[0]

        patch_attention_proc(pipe.unet, token_merge_args=token_merge_args)

        torch.manual_seed(seed)

        merged_img = pipe(prompt,
                          num_inference_steps=steps, height=height, width=width,
                          negative_prompt=negative_prompt,
                          guidance_scale=guidance_scale).images[0]

        base_img = base_img.convert("RGBA")
        merged_img = merged_img.convert("RGBA")
        merged_img = np.array(merged_img)
        halfh, halfw = height // 2, width // 2
        merged_img[halfh, halfw, 3] = 253 # set the center pixel of the merged image to be ever so slightly below 255 in alpha channel
        merged_img = Image.fromarray(merged_img)

        final_img = Image.new(size=(width * 2, height), mode="RGBA")

        if l_r > 0.5:
            left_img = base_img
            right_img = merged_img
        else:
            left_img = merged_img
            right_img = base_img

        final_img.paste(left_img, (0, 0))
        final_img.paste(right_img, (width, 0))

        which_image(final_img, width=width)

        return final_img


    gen.click(generate, inputs=[prompt, seed, steps, height, width, negative_prompt,
                                guidance_scale], outputs=[output_image])

demo.launch(share=True)