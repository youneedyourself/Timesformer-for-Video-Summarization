import numpy as np

def sum_of_squared_difference(vector1, vector2):
    squared_diff = np.square(vector1 - vector2)
    sum_squared_diff = np.sum(squared_diff)
    return sum_squared_diff

def tt01(features, threshold, num_frames):
    i = 0

    # list of clips (frames)
    clips = []
    for j in range(1, len(features)):
        if sum_of_squared_difference(features[i], features[j]) > threshold:
            clip = []
            for b in range(i*num_frames, j*num_frames):
                clip.append(b)

            random_num = round(len(clip)*0.15)
            random_Frames = sorted(random.sample(clip, random_num))
            #print(random_Frames)
            #print(f"{i} with {j}")
            i = j
            clips.extend(random_Frames)

    clip = []
    # Phần code sau khi for kết thúc
    #print(j)
    #print(i)
    if i==j:
        for c in range(j*num_frames, j*num_frames+num_frames):
            clip.append(c)
            random_num = round(len(clip)*0.15)
            random_Frames = sorted(random.sample(clip, random_num))
        #print("i == j")

    else: # (i<j)
        for c in range(i*num_frames, (j+1)*num_frames):
            clip.append(c)
            random_num = round(len(clip)*0.15)
            random_Frames = sorted(random.sample(clip, random_num))
        #print(f"{i} with {j}")
    clips.extend(random_Frames)
            
    return clips

def run():
    print("Starting...")

    model, processor, device = load_model()
    video = preprocess(video)

    print("Finished.")

if __name__ == "__main__":
    run()
    tt01()
    print("Done.")