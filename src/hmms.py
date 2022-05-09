#!/usr/bin/env python3

# %%

import random
import librosa
import numpy as np
import os
from hmmlearn import hmm

# be reproducible...
np.random.seed(1337)

# ---%<------------------------------------------------------------------------
# Part 1: Basics

# version 1.0.10 has 10 digits, spoken 50 times by 6 speakers
digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
nr = 50
speakers = list(['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'])

# %%
def load_fts(digit: int, spk: str, n: int):
    # load sounds file, compute MFCC; eg. n_mfcc=13
    path = os.path.join('../res/recordings/', f'{digit}_{spk}_{n}.wav')
    print(path)
    signal, sr = librosa.load(path)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13).T
    
    # standartization
    mfcc = (mfcc - mfcc.mean()) / (mfcc.std())
    
    return mfcc

# load data files and extract features
mfccs = {}
for speaker in speakers:
    for digit in digits:
        for n in range(nr):
            obs = load_fts(digit=digit, spk=speaker, n=n)
            mfccs[f'{digit}_{speaker}_{n}'] = obs   

# %% 

# implement a 6-fold cross-validation (x/v) loop so that each speaker acts as
# test speaker while the others are used for training
hmms = dict()

def get_linear_transmat(n_components: int):
    m1 = np.eye(n_components, k=0)
    m2 = np.eye(n_components, k=1)
    m = np.add(m1, m2)
    m = m / 2
    m[-1][-1] = 1.0
    return m

for speaker in speakers:
    test_speaker = speaker
    train_speakers = speakers.copy()
    train_speakers.remove(speaker)

# allocate and initialize the HMMs, one for each digit; set a linear topology
# choose and a meaningful number of states
# note: you may find that one or more HMMs are performing particularly bad;
# what could be the reason and how to mitigate that?

    for d in digits:

        n_components = 10
        model = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', init_params='cm', params='cmt')
        startprob = [1.0]
        startprob.extend([0.0 for i in range(n_components-1)])
        model.startprob_ = np.array(startprob)
        model.transmat_ = get_linear_transmat(n_components=n_components)
        #print(model.transmat_)

        hmms[d] = model


# train the HMMs using the fit method; data needs to be concatenated,
# see https://github.com/hmmlearn/hmmlearn/blob/38b3cece4a6297e978a204099ae6a0a99555ec01/lib/hmmlearn/base.py#L439
    train_data = dict()
    for d in digits:
        train_data[d] = {'X': np.array([]), 'lenghts': []}

    for key, observation in mfccs.items():
        digit, tmp_speaker, n = key.split('_')
        digit = int(digit)
        n = int(n)

        if tmp_speaker in train_speakers:
            if train_data[digit]['X'].size == 0:
                train_data[digit]['X'] = observation
            else:
                train_data[digit]['X'] = np.concatenate((train_data[digit]['X'], observation))
            train_data[digit]['lenghts'].append(len(observation))

    for d in digits:
        print(f'training model for {d}')
        #print(train_data[d]['X'].shape)
        #print(train_data[d]['lenghts'])
        #print(len(train_data[d]['lenghts']))
        hmms[d].fit(X=train_data[d]['X'], lengths=train_data[d]['lenghts'])
        #print(hmms[d].transmat_)
        #print(model.sample(20))
    break

# evaluate the trained models on the test speaker; how do you decide which word
# was spoken?

# compute and display the confusion matrix
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

# %%
# display the overall confusion matrix

# ---%<------------------------------------------------------------------------
# Part 2: Decoding

# generate test sequences; retain both digits (for later evaluation) and
# features (for actual decoding)
def generate_digit_sequenze(seq: list, speaker: str = ''):
    sequenze = np.array([])
    lengths = []

    for i in seq:
        if not speaker:
            spk = random.choice(speakers)
        else:
            spk = speaker
        key = f'{i}_{spk}_{random.randrange(0,50)}'
        print(key)
        if sequenze.size == 0:
            sequenze = mfccs[key]
        else:    
            sequenze = np.concatenate((sequenze, mfccs[key]))
        lengths.append(len(mfccs[key]))

    return sequenze, lengths

generated, lengths = generate_digit_sequenze(seq=[1,2,3], speaker='george')
print(generated.shape)
print(len(generated))
print(lengths)
print()
generated, lengths = generate_digit_sequenze(seq=[1,2,3])
print(generated.shape)
print(len(generated))
print(lengths)

# combine the (previously trained) per-digit HMMs into one large meta HMM; make
# sure to change the transition probabilities to allow transitions from one
# digit to any other
meta_model = hmm.GaussianHMM(n_components=10)

# use the `decode` function to get the most likely state sequence for the test
# sequences; re-map that to a sequence of digits

# use jiwer.wer to compute the word error rate between reference and decoded
# digit sequence

# compute overall WER (ie. over the cross-validation)

# ---%<------------------------------------------------------------------------
# Optional: Decoding


# %%
