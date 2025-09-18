# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import warnings
import copy
from decord import VideoReader, cpu
import numpy as np


pretrained_name = "StreamFormer/streamformer-llava-Qwen2.5-7B-Instruct-v1.5" # IMPORTANT: make sure to enable streaming_mode and set context_length in config.json
model_name = "llava_qwen"
device = "cuda"
device_map = "cuda"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained_name, None, model_name, device_map=device_map, torch_dtype="bfloat16")
model.eval()


conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

def load_video(video_path, max_frames_num,fps=1,force_sample=False):
    if max_frames_num == 0:
        return np.zeros((1, 224, 224, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    # import pdb;pdb.set_trace()
    return spare_frames,frame_time,video_time


video_path = "./test.mp4"
max_frames_num = 16
video,frame_time,video_time = load_video(video_path, max_frames_num, 1, force_sample=True)
video = image_processor.preprocess(video, return_tensors="pt")["pixel_values"].cuda().half()
video = [video]

question = DEFAULT_IMAGE_TOKEN + f"\nWhat is the person doing in the video?"
conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

cont = model.generate(
    input_ids,
    images=[video[0][:8]],
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print("#1 [Frames 0-7]:", text_outputs)

cont = model.generate(
    input_ids,
    images=[video[0][8:]],
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print("#2 [Frames 8-15 with context of 0-7 as kv-cache]:", text_outputs)

model.get_vision_tower().clear_cache() # clear kv-cache for streamformer

cont = model.generate(
    input_ids,
    images=[video[0]],
    modalities= ["video"],
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)[0].strip()
print("#3 [Frames 0-15]:", text_outputs) # should be the same as #2