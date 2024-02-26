import time
import spaces
import gradio as gr
import torch
import diffusers
from utils import patch_attention_proc, remove_patch
import math
import numpy as np
from PIL import Image
from threading import Semaphore

# Globals
css = """
h1 {
  text-align: center;
  display: block;
}
"""

# Pipeline
pipe = diffusers.StableDiffusionPipeline.from_pretrained("Lykon/DreamShaper").to("cuda", torch.float16)
pipe.scheduler = diffusers.EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.safety_checker = None

semaphore = Semaphore() # for preventing collisions of two simultaneous button presses

@spaces.GPU
def generate_baseline(prompt, seed, steps, height_width, negative_prompt, guidance_scale, method):

    semaphore.acquire()

    downsample_factor = 2
    ratio = 0.38
    merge_method = "downsample" if method == "todo" else "similarity"
    merge_tokens = "keys/values" if method == "todo" else "all"

    if height_width == 1024:
        downsample_factor = 2
        ratio = 0.75
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif height_width == 1536:
        downsample_factor = 3
        ratio = 0.89
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif height_width == 2048:
        downsample_factor = 4
        ratio = 0.9375
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0

    token_merge_args = {"ratio": ratio,
                "merge_tokens": merge_tokens,
                "merge_method": merge_method,
                "downsample_method": "nearest",
                "downsample_factor": downsample_factor,
                "timestep_threshold_switch": 0.0,
                "timestep_threshold_stop": 0.0,
                "downsample_factor_level_2": downsample_factor_level_2,
                "ratio_level_2": ratio_level_2
                }

    torch.manual_seed(seed)
    start_time_base = time.time()
    remove_patch(pipe)
    base_img = pipe(prompt,
                    num_inference_steps=steps, height=height_width, width=height_width,
                    negative_prompt=negative_prompt,
                    guidance_scale=guidance_scale).images[0]
    end_time_base = time.time()

    result = f"Baseline image: {end_time_base-start_time_base:.2f} sec"

    semaphore.release()

    return base_img, result


@spaces.GPU
def generate_merged(prompt, seed, steps, height_width, negative_prompt, guidance_scale, method):
    
    semaphore.acquire()

    downsample_factor = 2
    ratio = 0.38
    merge_method = "downsample" if method == "todo" else "similarity"
    merge_tokens = "keys/values" if method == "todo" else "all"

    if height_width == 1024:
        downsample_factor = 2
        ratio = 0.75
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif height_width == 1536:
        downsample_factor = 3
        ratio = 0.89
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0
    elif height_width == 2048:
        downsample_factor = 4
        ratio = 0.9375
        downsample_factor_level_2 = 1
        ratio_level_2 = 0.0

    token_merge_args = {"ratio": ratio,
                "merge_tokens": merge_tokens,
                "merge_method": merge_method,
                "downsample_method": "nearest",
                "downsample_factor": downsample_factor,
                "timestep_threshold_switch": 0.0,
                "timestep_threshold_stop": 0.0,
                "downsample_factor_level_2": downsample_factor_level_2,
                "ratio_level_2": ratio_level_2
                }

    patch_attention_proc(pipe.unet, token_merge_args=token_merge_args)
    torch.manual_seed(seed)
    start_time_merge = time.time()
    merged_img = pipe(prompt,
                        num_inference_steps=steps, height=height_width, width=height_width,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale).images[0]
    end_time_merge = time.time()

    result = f"{'ToDo' if method == 'todo' else 'ToMe'} image: {end_time_merge-start_time_merge:.2f} sec"

    semaphore.release()

    return merged_img, result



with gr.Blocks(css=css) as demo:
    gr.Markdown("# ToDo: Token Downsampling for Efficient Generation of High-Resolution Images")
    prompt = gr.Textbox(interactive=True, label="prompt")
    negative_prompt = gr.Textbox(interactive=True, label="negative_prompt")
    
    with gr.Row():
        method = gr.Dropdown(["todo", "tome"], value="todo", label="method", info="Choose Your Desired Method (Default: todo)")
        height_width = gr.Dropdown([1024, 1536, 2048], value=1024, label="height/width", info="Choose Your Desired Height/Width (Default: 1024)")

    with gr.Row():
        guidance_scale = gr.Number(label="guidance_scale", value=7.5, precision=1)
        steps = gr.Number(label="steps", value=20, precision=0)
        seed = gr.Number(label="seed", value=1, precision=0)

    with gr.Row():
        with gr.Column():
            base_result = gr.Textbox(label="Baseline Runtime")
            base_image = gr.Image(label=f"baseline_image", type="pil", interactive=False)
            gen = gr.Button("Generate Baseline")
            gen.click(generate_baseline, inputs=[prompt, seed, steps, height_width, negative_prompt,
                                        guidance_scale, method], outputs=[base_image, base_result])
        with gr.Column():
            output_result = gr.Textbox(label="Runtime")
            output_image = gr.Image(label=f"image", type="pil", interactive=False)
            gen = gr.Button("Generate")
            gen.click(generate_merged, inputs=[prompt, seed, steps, height_width, negative_prompt,
                                        guidance_scale, method], outputs=[output_image, output_result])

demo.launch(share=True)