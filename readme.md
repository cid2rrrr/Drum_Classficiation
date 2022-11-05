Drum kits Classificaion
===============

**Notice** Since I reset my Desktop w/o any backup, I can't update anymore

----
</br>

# Purpose

Unlike other instruments, Drum is the instrument that brings together multiple percussive instrumets, such as Kick, Snare, Hi-Hat, etc..  
Which means Learning how to classify drum kits using ML will improve your understanding of sound and ML. Of course it's more interesting than classifying MNIST dataset!


In this Project, I'm gonna classify seven drum kits in four major ways.


* Simple Neural Network  
* Convolution Neral Network  
* Bi-Directional LSTM  
* scikit-learn based Maschine Learning  

</br></br>

# Dataset

because of copyright issues, I cannot share the `*.wav`  file.  
You can buy some sample packs from the webpage like [Drum Broker][drumbroker_link] or [Samples From Mars][sfm_link].

[drumbroker_link]: https://hiphopdrumsamples.com "The Drum Broker"
[sfm_link]: https://samplesfrommars.com "Samples from Mars"
You can also download some samples for free from [FreeWaveSamples][fws_link] or [Drumkits Reddit][reddit_link].

[fws_link]: https://freewavesamples.com "Free Wave Sampels"
[reddit_link]: https://www.reddit.com/r/Drumkits/ "r/Drumkits"

the Drum kits are divided into seven major claaes, and collected 100 samples for each of the seven classes.

* Kick
* Snare
* Hi-Hat (Closed Hat)
* Shaker
* Open Hat & Crash (Cymbals)
* Tom
* Conga & Bongo


</br>

# Audio Augmentation

I didn't try the Audio Augmentation because I didn't know how to augment audio data at that time, but for those of you interested in related research, I leave this:

There are may ways to transform audio data.
EQ bands, Compressor, Reverb, Limiter, Pitch Shift(not recommended for this proj), Adding Noise, Freq. Masking(alos not recommended for this proj), etc..

It would be best done by hand, but we don't have much time, so I recommend installing two [Python][pedal_lnk] [Libraries][adm_lnk].

[pedal_lnk]: https://github.com/spotify/pedalboard "Spotify PedalBoard"

[adm_lnk]: https://github.com/iver56/audiomentations "Audiomentations"

[AddBackgroundNoise][b_lnk], [AddGaussianNoise][g_lnk], [Compressor][cp_lnk], and [BitCrush][c_lnk] are what I recommend for this project.

[b_lnk]: https://iver56.github.io/audiomentations/waveform_transforms/add_background_noise/ "Audiomentation Documentation"
[g_lnk]: https://iver56.github.io/audiomentations/waveform_transforms/add_gaussian_noise/ "Audiomentation Documentation"
[c_lnk]: https://github.com/spotify/pedalboard/blob/master/pedalboard/plugins/Bitcrush.h "Pedalboard BitCrush.h"
[cp_lnk]: https://github.com/spotify/pedalboard/blob/master/pedalboard/plugins/Compressor.h "Pedalboard Compressor.h"


</br></br></br>

# 0. Common Preprocess

The Volume and Lenght of the samples collected are all different. Even some samples have a slight delay before onset.  

To Remove the delay, `librosa.effects.trim` is used.  
And to normalize the sounds, `librosa.util.normalize` is used.  
Changing the order in which the effects are applied will change the results, but it doesn't matter much.  

Finally, the ratio of training data to evaluation data is divided into 8:2.


</br></br>

# 1. Simple Neural Network w/ waveform (waveform.ipynb)

### What is Neural Network : [wiki][nn_wiki]

[nn_wiki]: https://en.wikipedia.org/wiki/Neural_network "Neural network wikipedia"

It uses the simples model of the four methods.

Even creating a smiple model with one relu layer, you can get a not bad result.

In order to find optimized performance while increasing the complexity of the model, the generating model method is overriden.
Wrapping the method with `keras.wrappers.scikit_learn.KerasRegressor`, you can find the best parameters through the RandomSearchCV from scikit-learn.


</br></br>

# 2. Convolution Neural Network w/ Mel-Spectrogram

## Mel-Spectrogram

### STFT(SHort Time Fouriter Transformer)  
STFT is a method to analyze the sound whose frequency characteristics vary with time. It is divided into time series constant time interval and the spectrum is obtained for each interval.

### Mel-Spectrogram
 Spectrum that changed the unit of frequency to mel unit according to mel-scale


## Convolutional Neural Networ
It is the one of the deep learning structures created by mimicking the human optic nerve.  
It maintains the spatial information of the image using the convolution operation and dramatically reduces the computational complexity compared to the general neural network, and performs well in image classification.

### More Detail about [CNN][conv_lnk]  /  [Mel-Spectrogram][mel_lnk]

[conv_lnk]: https://cs231n.github.io/convolutional-networks/

[mel_lnk]: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53 "medium.com Understanding the Mel Spectrogram"

</br>

The Mel-Spectrogram array is extracted using `librosa` library.
On this project, the y-axis(frequency) is divided into 128 equal parts and the x-axis(time) is fixed to 20.

A simple model consisting of three layers of convolution and max pooling.
It is a simple model, but overall it showed the best performance.

If you define the model generation function well, you can get better results using the `KerasRegressor`.  
There is also a way to create a 3-channel image file using the `matplotlib.pyplot` library and then learn it using a well-defined model such as ResNet or LeNet.


</br></br>

# 3. scikit-learn w/ Spectral Feature

On `extract.ipynb`, Five features of each data were extracted by unit time through the Python `librosa` library.

The extracted features ar as follow:
* Duration
    * the length of the data after trimming

* Spectral Bandwidth
    * the difference between the upper and lower frequencies

* Spectral Rolloff
    * It can be defined as the action of a specific type of filter which is designed to roll off the frequencies outside to a specific range.  
    the default value of roll_percent is 0.85, but 0.8 is used in this project

* Spectral Flatness
    * the ratio of the geometric mean to the arithmetic mean of a power spectrum.

* Zero Crossing Rate
    * the measure of the rate at which the signal is going through the zeroth line.


### More detail about Spectral features[[1]][roll_lnk] [[2]][sp_lnk]

</br>

[roll_lnk]: https://www.johndcook.com/blog/2016/05/03/spectral-flatness/ "Spectral Flatness"

[sp_lnk]: https://analyticsindiamag.com/a-tutorial-on-spectral-feature-extraction-for-audio-analytics/ "Others"


A total of 57 feature values (14 values per one feature, and duration) are extracted. 

A total of 11 models provided by `scikit-learn` such as decision tree, random forest, SVC, etc. were trained and evaluated.

Overall, it showed an accuracy of more than 70%.
In addition, Visualizing one of the decision trees in random forest, it can be seen that Spectrall rolloff values are most frequently used for decision making.

</br></br>

# 4. Bi-directional LTSM using waveform

Sound is a vibration of air, which means that it can be said to be a time series data.
Thus, RNN model can be used for classifing.

Recurrent Neural Network is a class of neural networks that allow previous outputs to be used as inputs while having hidden states.

However, there is a disadvantage that the information of the front RNN layer is not well learned because only the output hidden state of the RNN layer for the last time step is transmitted to the next layer.

In order to improve the limitations of the existing model, a bi-directional RNN model is proposed to combine the previous part of the sequence with the subsequent part.

### More detail about [RNN][rnn_lnk] / [Bi-Directional RNN][ls_lnk]

[rnn_lnk]: https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-recurrent-neural-networks "RNN stanford edu"

[ls_lnk]: https://d2l.ai/chapter_recurrent-modern/bi-rnn.html "bi-directional rnn"

</br>

The performance of the classification model is not good because the length of the waveform itself is so long.
Simply replacing the mel-spectrogram with the input instead of the waveform will provide a significant performance improvement.
Due to data loss, it is not applied in this project.


</br></br></br></br>

# Future work

It would be an interesting approach not only to change the input shape or use new models such as transformers, but also to try multiple label classification for synthesized sounds, or to classify them into synth drums and analog drums.








