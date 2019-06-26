import random

POSITIVE_OPLEX = ["good", "lovely", "excellent", "fortunate", "pleasant", "delightful", "perfect", "loved", "love", "happy",
                 "admirable", "unselfish", "beauty", "outstandingly", "wisdom" ,"encourage" ,"effective", "amusing", "recommended", "best"]
NEGATIVE_OPLEX = ["bad", "horrible", "poor",  "unfortunate", "unpleasant", "disgusting", "evil", "hated", "hate", "unhappy",
                 "hurt", "tired", "decry", "addict", "disdainfully", "penalize", "fail", "losing", "pest", "jerk"]

ENTITY_WIKI = ['he', 'she', 'American', 'family', 'Government', 'son', 'athlete', 'students', 'military', 'Buddhist',
              'Catholic', 'army', 'actress', 'Turkish', 'Executive', 'soldier', 'doctor', 'undergraduate', 'manufacturers', 'children']
NOTITY_WIKI = ['the', 'of', 'their', 'time', 'first', 'been', 'part', 'can', 'United', 'school',
               'diesel', 'creates', 'Punjab', 'o', 'Grove', 'celebrate', 'Liberty', 'teeth', 'acoustic', 'virtually']

def sent_seeds(num=10):
    return POSITIVE_OPLEX[:num], NEGATIVE_OPLEX[:num]

def entity_seeds(num=10):
    return ENTITY_WIKI[:num], NOTITY_WIKI[:num]

def random_seeds(words, lexicon, num):
    sample_set = list(set(words).intersection(lexicon))
    seeds = random.sample(sample_set, num)
    return [s for s in seeds if lexicon[s] == 1], [s for s in seeds if lexicon[s] == -1]