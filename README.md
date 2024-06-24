# Timesformer-for-Video-Summarization

![Python version](https://img.shields.io/badge/python-3.8.0-blue)
![License](https://img.shields.io/badge/license-MIT-white)

Learn Timesformer and apply it to video summarization using two unsupervised methods: K-means and Sum of Squared Difference.

# Installation.

1. Clone the repository: 

```
git clone https://github.com/youneedyourself/Timesformer-for-Video-Summarization.git
```

2. Install the required dependencies:

```
pip install -r requirements.txt
```

## Dataset

### SumMe
The data can be downloaded from the [project page](https://gyglim.github.io/me/vsum/index.html).
Copy the files in `GT/` and videos `videos/` in SumMe to `data/summe/GT/` and `data/summe/videos/`.
Like this:

```
data/summe
├── GT/
│   ├── video1.mat
│   ├── video.mat
│   └── ...
├── videos/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── ...
```


## Methods

1. Visualize attention map

Please go to this link: https://github.com/yiyixuxu/TimeSformer-rolled-attention and clone this repo to your computer, after that, use 2 notebooks in my repo to visualize the attention map of Timesformer.

2. Video summarization:
- Kmeans
```
python src/kmeans.py
```

- Sum of Squared Diff 01
```
python src/tt01.py
```
- Sum of Squared Diff 02
```
python src/tt02.py
```

3. Evaluation

Please go to this link: https://github.com/mayu-ot/rethinking-evs and clone this repo to your computer. Follow the repo instruction, use 3 notebooks in my repo to create a json file to evaluate the summarized videos.

## Deploying

I'm also deployed the summarize system by using Gradio and Hugging Face, enter https://huggingface.co/spaces/namnh2002/video-summarization_timesformer to try it yourself.

Source code: https://github.com/youneedyourself/Video-Summarization_Timesformer

## References

    - Video summarizer - https://github.com/AmitDeo/video-summarizer
    - Attention Rollout - https://github.com/yiyixuxu/TimeSformer-rolled-attention
    - Rethinking the Evaluation of Video Summaries  - https://github.com/mayu-ot/rethinking-evs
    - Timesformer - Is Space-Time Attention All You Need for Video Understanding?