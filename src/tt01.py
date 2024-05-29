import numpy as np
from utils import * 
import random

def tt01(features, threshold):

    i = 0
    clips = []

    # compare the sum of squared difference between clips i and j
    for j in range(1, len(features)):
        if sum_of_squared_difference(features[i], features[j]) > threshold:
            clip = []

            # add frames from clip i to j-1 to the clip list
            for b in range(i*8, j*8):
                clip.append(b)

            # randomly select 15% of the frames from the clip list
            random_num = round(len(clip)*0.15)

            # sort the frames in the clip list to ensure the order of the frames
            random_Frames = sorted(random.sample(clip, random_num))
            i = j
            clips.extend(random_Frames)

    # add the last clip to the clip list
    clip = []
    if i==j:
        for c in range(j*8, j*8+8):
            clip.append(c)
            random_num = round(len(clip)*0.15)
            random_Frames = sorted(random.sample(clip, random_num))
        #print("i == j")

    else: # (i<j)
        for c in range(i*8, (j+1)*8):
            clip.append(c)
            random_num = round(len(clip)*0.15)
            random_Frames = sorted(random.sample(clip, random_num))
        #print(f"{i} with {j}")

    clips.extend(random_Frames)
            
    return clips

def run():
    print("Starting... - using TT01")

    #load model, processor, and device
    model, processor, device = load_model()

    #get the list of videos path
    videos_path = get_list_videos_path()
    
    # set the output path
    OUTPUT_VIDEO_PATH = "data/summarized/tt01/videos"
    OUTPUT_TXT_PATH = "data/summarized/tt01/txts"
    threshold = 400

    for video in videos_path:
        print(f"Summarizing the video {video}.")

        #preprocess the video
        frames, features, video_fps, clip_sample_rate, _ = preprocess(video, model, processor, device)

        # tt01 algorithm
        selected_frames = tt01(features, threshold)

        #save the output video and txt
        video_name = video.split("/")[-1]
        output_video = os.path.join(OUTPUT_VIDEO_PATH, f"summarized_{video_name}")
        to_video(selected_frames, frames, output_video, video_fps)

        output_txt = os.path.join(OUTPUT_TXT_PATH, f"{video_name}.txt")
        to_txt(selected_frames, output_txt, clip_sample_rate)
        print(f"Finished summarizing the video {video}.")

    print("All done.")

def test_1_sample():
    print("Starting... - using TT01")

    model, processor, device = load_model()
    video_path = "data/summe/videos/jumps.mp4"
    threshold = 400

    frames, features, video_fps, clip_sample_rate, _ = preprocess(video_path, model, processor, device)
    print("Number of frames: ", len(frames))
    print("Shape of each frame: ", frames[0].shape)
    print("Number of features: ", len(features))
    print("Shape of each feature: ", features[0].shape)

    selected_frames = tt01(features, threshold)

    video_name = video_path.split("/")[-1]

    output_video = os.path.join("data/summarized/tt01/videos", f"summarized_{video_name}")
    to_video(selected_frames, frames, output_video, video_fps)

    output_txt = os.path.join("data/summarized/tt01/txts", f"{video_name}.txt")
    to_txt(selected_frames, output_txt, clip_sample_rate)
    print(f"Finished summarizing the video {video_path}.")

if __name__ == "__main__":
    #run()
    test_1_sample()