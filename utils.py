import pickle

file = open("dentistry-tfidf-vectorizer.pickle", 'rb')
model = pickle.load(file)

def keyword_extractor(text, model):
    vocab = model.get_feature_names()
    tfidf_array = model.transform([text]).toarray()[0]
    tfidf_nonzero = tfidf_array.nonzero()[0]
    weights = []
    for index in tfidf_nonzero:
        word = vocab[index]
        tfidf = tfidf_array[index]
        if len(word.split(' ')) == 1:
            weights.append((word,tfidf))
    result = sorted(weights, key=lambda tup: tup[1], reverse=True)
    return result

def print_keywords(text):
    result = keyword_extractor(text,model)
    print("   keyword\tweight")
    for i,(word,weight) in enumerate(result):
        print('{}. {}\t{:.2f}'.format(i+1,word,weight))