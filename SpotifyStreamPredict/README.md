## SpotifyStreamPredict Introduction

This project was built to predict the total number of streams a song may get, based on Spotify's track audio features (i.e. valence, danceability) and other musical influences such as musical key or BPM.

## Methodology

Our dataset was sourced [here](https://www.kaggle.com/datasets/nelgiriyewithana/top-spotify-songs-2023), detailing a comprehensive list of the most famous songs in 2023. It additionally details insights into each songs attributes, Spotify track audio features, and more.

In order to generate predictions, we train a feed-forward neural network with a series of DropOut and Dense layers, using TensorFlow Keras. After training for 18 epochs, we achieve an MSE of 0.005.

## Findings

Evaluating the model on our test set, we range from **0%-17**. Our results indicate a strong performance on songs with smaller/medium ranged streams, only performing unfavorably for songs with extremely high popularity, indicating a weak generalization to outliers.