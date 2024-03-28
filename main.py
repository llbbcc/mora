import time
from PIL import Image
import openai
from moviepy.editor import VideoFileClip, concatenate_videoclips

import torch
from diffusers.utils import export_to_video
from diffusers import StableVideoDiffusionPipeline, DiffusionPipeline, StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

def prompt_enhance(prompt):
    openai.api_key = "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" 
    content = "Simply expand the following passage to make it fuller and more detailed. Return only to the expanded version." + prompt
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}]
    )
    return dict(completion)['choices'][0]['message']['content']

def get_last_frame(video_path, image_path):
    with VideoFileClip(video_path) as video:
        last_frame = video.get_frame(video.duration - 0.01)
    # TODO enhance the detail of the last_frame
    last_frame_image = Image.fromarray(last_frame)
    last_frame_image.save(image_path)
    return image_path

def main(prompt, opt='txt2vid', n_steps=40, num_iterations=3, video_path=None, cache_dir='/data/disk1/jupyter/watermark/huggingface/'):
    # model load
    base_pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
        force_download=True, resume_download=False,
    )
    pix_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pix_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pix_pipe.scheduler.config)
    svd_pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt", 
        cache_dir=cache_dir,
        torch_dtype=torch.float16,
        variant="fp16"
    )
    svd_pipe.enable_model_cpu_offload()
    base_pipe.to("cuda")
    pix_pipe.to('cuda')

    # prompt enhancement by GPT 3.5
    prompt_ = prompt_enhance(prompt)

    # text-to-image
    def txt2img(out_path='image.png'):
        image = base_pipe(
            prompt=prompt_,
            num_inference_steps=n_steps,
            height=576,
            width=1024,
        ).images[0]
        image.save(out_path)
    # image-to-image
    def img2img(img_path='image.png', out_path='image_refine.png'):
        img = Image.open(img_path).resize((1024, 576))
        image = pix_pipe(prompt, image=img, num_inference_steps=n_steps, image_guidance_scale=1).images[0]
        image.save(out_path)
    # image-to-video
    def img2vid(video_paths=None, image_path='image_refine.png', output_video="final_output_video.mp4"):
        if video_paths == None:
            video_paths = []
        for iteration in range(num_iterations):
            image = Image.open(image_path).resize((1024, 576))
            seed = int(time.time())
            torch.manual_seed(seed)
            frames = svd_pipe(image, decode_chunk_size=12, generator=torch.Generator(), motion_bucket_id=127).frames[0]
            video_path = f"video_segment_{iteration}.mp4"
            export_to_video(frames, video_path, fps=5)
            video_paths.append(video_path)
            image_path = get_last_frame(video_path, "last_frame.png")
        
        clips = [VideoFileClip(path) for path in video_paths]
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(output_video)

    if opt == 'txt2vid':
        txt2img()
        img2img()
        img2vid()
    elif opt == 'txtcond2video':
        img2img()
        img2vid()
    elif opt == 'exdvid':
        img_path = get_last_frame(video_path, 'last_frame.png')
        img2vid(video_paths=[video_path], image_path='last_frame.png', video_path='extended_video.mp4')
    elif opt == 'vid2vid':
        img_path = get_last_frame(video_path, 'last_frame.png')
        img2img(img_path)
        img2vid(output_video='edited_video.mp4')
    elif opt == 'simdigwrd':
        txt2img()
        img2vid()
    else:
        raise KeyError


if __name__ == '__main__':
    prompt = "A vibrant coral reef teeming with life under the crystal-clear blue ocean, with colorful fish swimming among the coral, rays of sunlight filtering through the water, and a gentle current moving the sea plants."
    main(prompt, opt='txt2vid', num_iterations=1)