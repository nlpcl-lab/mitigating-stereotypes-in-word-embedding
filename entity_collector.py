from nltk.corpus import wordnet as wn

car_synsets = wn.synsets('occupation')
print(car_synsets)


for car in car_synsets:
    print("lemmas: ", car.lemmas())
    print("definition: ", car.definition())
    print("hypernyms:", car.hypernyms())
    print("hyponyms:", car.hyponyms())
    print('-' * 40, '\n\n')

