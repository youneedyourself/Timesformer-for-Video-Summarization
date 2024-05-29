import faiss
from sklearn.metrics import pairwise_distances_argmin_min
from utils import * 

def kmeans(number_of_clusters, features):
    # Cluster the frames using K-Means

    # K-means from sklearn
    #kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(features)

    # K-means from faiss
    ncentroids = number_of_clusters
    niter = 10
    verbose = True
    x = features
    
    # Take the first dimension of the first element of the list
    dimension = x[0].shape[0]

    kmeans = faiss.Kmeans(dimension, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(x)

    #closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
    closest, _ = pairwise_distances_argmin_min(kmeans.centroids, x)
    
    closest_clips_frames = []

    for i in sorted(closest):
        for idx in range(i*8, (i+1)*8):
            closest_clips_frames.append(idx)

    return closest_clips_frames

def run():
    print("Starting... - using Kmeans")

    #load model, processor, and device
    model, processor, device = load_model()

    #get the list of videos path
    videos_path = get_list_videos_path()
    print(videos_path)
    
    # set the output path
    OUTPUT_VIDEO_PATH = "data/summarized/kmeans/videos"
    OUTPUT_TXT_PATH = "data/summarized/kmeans/txts"

    for video in videos_path:
        print(f"Summarizing the video {video}.")

        #preprocess the video
        frames, features, video_fps, clip_sample_rate, n_cluster = preprocess(video, model, processor, device)

        #kmeans clustering
        selected_frames = kmeans(n_cluster, features)

        #save the output video and txt
        video_name = video.split("/")[-1]
        output_video = os.path.join(OUTPUT_VIDEO_PATH, f"summarized_{video_name}")
        to_video(selected_frames, frames, output_video, video_fps)

        output_txt = os.path.join(OUTPUT_TXT_PATH, f"{video_name}.txt")
        to_txt(selected_frames, output_txt, clip_sample_rate)
        print(f"Finished summarizing the video {video}.")

    print("All done.")

# in case you want to test the code with only one sample
def test_1_sample():
    print("Starting... - using Kmeans")

    model, processor, device = load_model()
    video_path = "data/summe/videos/jumps.mp4"

    frames, features, video_fps, clip_sample_rate, n_cluster = preprocess(video_path, model, processor, device)
    print("Number of frames: ", len(frames))
    print("Shape of each frame: ", frames[0].shape)
    print("Number of features: ", len(features))
    print("Shape of each feature: ", features[0].shape)
    
    selected_frames = kmeans(n_cluster, features)
    
    video_name = video_path.split("/")[-1]
    
    output_video = os.path.join("data/summarized/kmeans/videos", f"summarized_{video_name}")
    to_video(selected_frames, frames, output_video, video_fps)

    output_txt = os.path.join("data/summarized/kmeans/txts", f"{video_name}.txt")
    to_txt(selected_frames, output_txt, clip_sample_rate)
    print(f"Finished summarizing the video {video_path}.")


if __name__ == "__main__":
    #run()
    test_1_sample()