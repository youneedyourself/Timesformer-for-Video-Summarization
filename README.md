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

# Dataset

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

# Project Organization
--------------------

    .
    ├── LICENSE
    ├── README.md
    ├── data
    │   ├── interim
    │   ├── processed
    │   └── raw # please see "Data" description above
    │        ├── summe
    │        │     └── GT/ 
    │        │     └── shot_SumMe.mat
    │        ├── tvsum
    │        │     └── ydata-tvsum50.mat
    │        │     └── shot_TVSum.mat
    │        └── example.json
    ├── notebooks
    └── src

# Methods

- visualize attention map

- algorithm:
    - kmeans
    - tt01
    - tt02
```
python src/kmeans.py
```
- evaluation

# Note

i'm using clip_Sample_rate 1 -> if you want to increase, pls do it yourself

# Results


## References

    - video summarizer
    - visualize
    - evaluation
    - timesformer
    - summer