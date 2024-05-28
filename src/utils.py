from transformers import TimesformerModel, VideoMAEImageProcessor
import torch
import cv2
import numpy as np
from torchvision.transforms import Lambda
from pytorchvideo.transforms import (
    Normalize,
)
from torchvision.transforms import (
    Lambda,
)
import os
from os.path import isfile, join, basename

def extract_features(frames, device, model, image_processor):
    # Convert frames to tensor
    frames_tensor = torch.stack([torch.from_numpy(frame) for frame in frames])
    # Change the order of the tensor to (num_frames, channel, height, width)
    frames_tensor = frames_tensor.permute(3, 0, 1, 2).to(device)

    # Get the mean and std of the image processor
    mean = image_processor.image_mean
    std = image_processor.image_std

    # Normalize frames
    frames_tensor = Lambda(lambda x: x / 255.0)(frames_tensor)
    frames_tensor = Normalize(mean, std)(frames_tensor)

    # Change the order of the tensor to (num_frames, channel, height, width) and add a batch dimension
    frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0)

    # Load the model to the device
    model.to(device)
    model.eval()
    outputs = model(frames_tensor)

    # Get the output after the Transformer Encoder (MLP head)
    final_output = outputs[0][:, 0]

    return final_output

def preprocess(video_path, model, processor, device):
    
    # Find the size to resize
    if "shortest_edge" in processor.size:
        height = width = processor.size["shortest_edge"]
    else:
        height = processor.size["height"]
        width = processor.size["width"] 
    resize_to = (height, width)

    video = cv2.VideoCapture(video_path)
    # Total number of frames in the video
    length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # F/Fs
    clip_sample_rate = 1

    # FPS
    video_fps = video.get(cv2.CAP_PROP_FPS)  

    # F
    num_frames = 8

    frames = []
    features = []

    for i in range(length):
        ret, frame = video.read()
        if not ret:
            break
        # Remember to resize the frame, it will crash the cpu if you don't
        frame = cv2.resize(frame, resize_to)
        frames.append(frame)

    # Find key frames by selecting frames with clip_sample_rate
    key_frames = frames[::clip_sample_rate] 
    #print('total of frames after sample:', len(selected_frames))

    # Remove redundant frames to make the number of frames can be divided by num_frames
    num_redudant_frames = len(key_frames) - (len(key_frames) % num_frames)

    # Final key frames
    final_key_frames = key_frames[:num_redudant_frames]
    #print('total of frames after remove redundant frames:', len(selected_frames))

    for i in range(0, len(final_key_frames), num_frames):
        if i % num_frames*50 == 0:
            print(f"Loading {i}/{len(final_key_frames)}")
        
        # Input clip to the model
        input_frames = final_key_frames[i:i+num_frames]
        # Extract features
        batch_features = extract_features(input_frames, device, model, processor)
        # Convert to numpy array to decrease the memory usage
        batch_features = np.array(batch_features.cpu().detach().numpy())
        features.extend(batch_features)

    number_of_clusters = round(len(features)*0.15)

    return final_key_frames, features, video_fps, clip_sample_rate, number_of_clusters
    
def to_video(selected_frames, frames, output_path, video_fps):
    
    print("MP4 Format.")
    # Write the selected frames to a video
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (frames[0].shape[1], frames[0].shape[0]))

    # selected_frames is a list of indices of frames
    for idx in selected_frames:
        video_writer.write(frames[idx])
    
    video_writer.release()
    print("Completed summarizing the video (wait for a moment to load).")

def to_txt(selected_frames, output_path, clip_sample_rate):
    # Write the selected frames to a txt file

    with open(output_path, "w") as file:
        for item in selected_frames:
            file.write(str(item) + "\n")
    
    print("Completed summarizing the txt (wait for a moment to load).")

def load_model():
    try:
        model=TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k600")
        processor=VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using {device}")
        return model, processor, device
    
    except Exception as e:
        print(e)

def sum_of_squared_difference(vector1, vector2):
    squared_diff = np.square(vector1 - vector2)
    sum_squared_diff = np.sum(squared_diff)
    return sum_squared_diff

def get_list_videos_path():
    VIDEO_DIR = "data/summe/videos"

    list_videos_path = []
    for video in os.listdir(VIDEO_DIR):
        if not video.endswith(".mp4"):
            continue
        list_videos_path.append(os.path.join(VIDEO_DIR, video))

    return list_videos_path