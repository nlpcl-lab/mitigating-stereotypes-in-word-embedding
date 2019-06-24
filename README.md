Word embedding with mitigating stereotypes through sentiment modulation
================================================
We provide the code and data for the following paper: Mitigating Stereotypes in Word Embedding through Sentiment Modulation by Huije Lee, Jin-Woo Chung, Jong C. Park. Korea Software Congress 2018 (KSC 2018), Pyeongchang (Korea), December 2018. This repository provides a model that mitigates stereotypes in word embedding through sentiment modulation. 

### Requirements
- Python >= 3.5
- Tensorflow >= 1.10.0
- scikit-learn >=	0.19.1
- keras == 0.3.3

## Get a word embedding with mitigating stereotypes
### prerequistes
* Original word embedding for mitigating stereotype (e.g. [word2vec google-news vectors](https://github.com/mmihaltz/word2vec-GoogleNews-vectors) with binary=False)
* Opinion Lexicon
  - Note that all external resources should be located at ```source/```

### code
- Prepare an original word embedding
- ```python mitigating_stereotypes.py``` generate a word embedding with mitigating stereotypes


