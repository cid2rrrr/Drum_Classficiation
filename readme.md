Unlike other instruments, Drum is the instrument that brings together multiple percussive instrumets, such as Kick, Snare, Hi-Hat, etc..
Which means Learning how to classify drum kits using ML will improve your understanding of sound and ML. Of course it's more interesting than classifying MNIST dataset!


In this Project, I'm gonna classify seven drum kits in four major ways.


Simple Neural Network
Convolution Neral Network
Bi-Directional LSTM
scikit-learn based Maschine Learning

Dataset

because of copyright issues, I cannot share the `*.wav`  file.
You can buy some sample packs from the webpage like [Drum Broker][drumbroker_link] or [Samples From Mars][sfm_link].

[drumbroker_link]: https://hiphopdrumsamples.com "The Drum Broker"
[sfm_link]: https://samplesfrommars.com "Samples from Mars"
You can also download some samples for free from [FreeWaveSamples][fws_link] or [Drumkits Reddit][reddit_link].

[fws_link]: https://freewavesamples.com "Free Wave Sampels"
[reddit_link]: https://www.reddit.com/r/Drumkits/ "r/Drumkits"

I divided the Drum kits into seven major claaes, and collected 100 samples for each of the seven classes.

Kick
Snare
Hi-Hat (Closed Hat)
Shaker
Open Hat & Crash (Cymbals)
Tom
Conga & Bongo


Audio Augmentation

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


0. Common Preprocess

The Volume and Lenght of the samples collected are all different. Even some samples have a slight delay before onset.

To Remove the delay, librosa.effects.trim is used.
And to normalize the sounds, librosa.util.normalize is used.
Changing the order in which the effects are applied will change the results, but it doesn't matter much.




1. Simple Neural Network w/ waveform (waveform.ipynb)

What is Neural Network : [wiki][nn_wiki]

[nn_wiki]: https://en.wikipedia.org/wiki/Neural_network "Neural network wikipedia"

It uses the simples model of the four methods.

Even creating a smiple model with one relu layer, you can get a not bad result.

In order to find optimized performance while increasing the complexity of the model, the generating model method is overriden.
Wrapping the method with keras.wrappers.scikit_learn.KerasRegressor, you can find the best parameters through the RandomSearchCV from scikit-learn.



2. Convolution Neural Network w/ Mel-Spectrogram

Mel-Spectrogram
    STFT(SHort Time Fouriter Transformer)
    STFT is a method to analyze the sound whose frequency characteristics vary with time. It is divided into time series constant time interval and the spectrum is obtained for each interval.

    MelSpectrogram
    Spectrum that changed the unit of frequency to mel unit according to mel-scale

More Detail [Link][mel_lnk]

[mel_lnk]: https://medium.com/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53 "medium.com Understanding the Mel Spectrogram"

CNN
    is the one of the deep learning structures created by mimicking the human optic nerve.


