This is an unofficial implementation of "Mora: Enabling Generalist Video Generation via A Multi-Agent Framework".

We use GPT-3.5-turbo to enhance the prompt.

We use models [SDXL](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0), [InstructPix2Pix](https://huggingface.co/timbrooks/instruct-pix2pix), and [SVD](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) to realize functions text-to-image, image-to-image, and image-to-video, respectively.

For arguments of `main`:
```
prompt: description of the scene.
opt: select the task. choice=['txt2vid', 'txtcond2video', 'exdvid', 'vid2vid', 'simdigwrd'].
n_steps: time steps for SDXL and InstructPix2Pix.
num_iterations: the length of the generated video is 5*num_iterations. 
video_path: the path of the video for video extension or video editing.
cache_dir: save the pre-trained model.
```

You can control the length of each generated video by modifying the `fps` of line 81. The total length of the generated video is `25*num_iterations/fps`
