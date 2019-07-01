import config, embedding

WORD_EMBEDDING_CONVERT_NAME = config.MODEL_DIR + config.MODEL_NAME + ".w2v.300d.txt"
CONVERT_BINARY_TO_TXT = True

if __name__ == "__main__":
    print("corpus: {} / consider_gender: {}".format(config.MODEL_NAME, config.CONSIDER_GENDER))

    # 1. training w2v
    w2v = embedding.W2vModel()
    if CONVERT_BINARY_TO_TXT:
        w2v.save(WORD_EMBEDDING_CONVERT_NAME)