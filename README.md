Word embedding with mitigating stereotypes through sentiment modulation
================================================
We provide the code and data for the following paper: Mitigating Stereotypes in Word Embedding through Sentiment Modulation by Huije Lee, Jin-Woo Chung, Jong C. Park. Korea Software Congress 2018 (KSC 2018), Pyeongchang (Korea), December 2018. This repository provides a model that mitigates stereotypes in word embedding through sentiment modulation. 

### Requirements
- Python >= 3.5
- Tensorflow >= 1.10.0
- scikit-learn >=	0.19.1
- keras == 0.3.3

### prerequistes
* Download IBM Debater® - Claims and Evidence dataset. [[here]](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml)
* Download Glove word embedding. [[here]](https://nlp.stanford.edu/projects/glove/)
* Install StanfordCoreNLP. [[here]](https://stanfordnlp.github.io/CoreNLP/index.html)
* Download [UCI Adult dataset](https://archive.ics.uci.edu/ml/datasets/Adult) (only for test) 
Note that all external resources should be located at ```source/```

### Usage
#### Get a word embedding with mitigating stereotypes
- ```python base_embeddings.py``` train word embedding -> make 'w2v_wiki_sg_300_neg5_it2.model'
- ```python evaluate_methods.py``` train word embedding with mitigating stereotypes -> make 'my_embedding_wikiSEED_NUM'
  run evaluate_methods.py (SEED_NUM, MODEL_NAME, MY_MODEL_NAME, VOCAB_LIMIT) -> make 'my_embedding_wikiSEED_NUM'
  
#### Test our word embedding
- ```python base_embeddings.py``` train word embedding -> make 'w2v_wiki_sg_300_neg5_it2.model' (baseline)
- ```python evaluate_methods.py``` train word embedding with mitigating stereotypes -> make 'my_embedding_wikiSEED_NUM' (transformed)
  please note sentiment, entity cutoff with space_order
- ```python base_embeddings.py``` show statistics (compared to other models)
  before running, set sentiment, entity cutoff with space_order -> 
  ```my = MyModel(threshold=<entity cutoff>, space_order=[<sent order>, <entity order>]```

### Note
- Model parameter files and required data for this module are available at http://credon.kaist.ac.kr/downloads

### Reference
* William L. Hamilton, Kevin Clark, Jure Leskovec, and Dan Jurafsky. Inducing Domain-Specific Sentiment Lexicons from Unlabeled Corpora. Proceedings of EMNLP. 2016. (to appear; arXiv:1606.02820) [[site](https://github.com/williamleif/socialsent)]
* Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings by Tolga Bolukbasi, Kai-Wei Chang, James Zou, Venkatesh Saligrama, and Adam Kalai. Proceedings of NIPS 2016. [[site](https://github.com/tolga-b/debiaswe)]

