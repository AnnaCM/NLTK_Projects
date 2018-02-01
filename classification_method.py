import nltk
import random, pickle
from nltk.corpus import stopwords
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from nltk.tokenize import word_tokenize
from statistics import mode

class Classification:
    def __init__(self, classifier):
        self.classifier = list(classifier)

    def classify(self, features):
        records = []
        for c in self.classifier:
            records.append(c.classify(features))
        return mode(records)

save_emails_f = open("pickled_files/all_emails.pickle","rb")
all_emails = pickle.load(save_emails_f)
save_emails_f.close()

save_common_words_f = open("pickled_files/common_words.pickle","rb")
common_words = pickle.load(save_common_words_f)
save_common_words_f.close()

type_of_words = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','MD','LS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
eng_stop_words = set(stopwords.words('english'))

def feat(emails):
    words_in_emails = word_tokenize(emails)
    tagged = nltk.pos_tag(words_in_emails)
    part_words = []
    for w,t in tagged:
        if w not in eng_stop_words and t in type_of_words:
            part_words.append(w.lower())
    lst = {}
    for w in common_words:
        lst[w] = (w in part_words)

    return lst

save_featuresets_f = open("pickled_files/featuresets.pickle","rb")
featuresets = pickle.load(save_featuresets_f)
save_featuresets_f.close()

random.shuffle(featuresets)
training_set = featuresets[:27000] #80% of featuresets
testing_set = featuresets[27000:] #20% of featuresets

save_classifier_f = open("pickled_files/naivebayes.pickle","rb")
classifier = pickle.load(save_classifier_f)
save_classifier_f.close()

save_classifier_f = open("pickled_files/MultinomialNB_classifier.pickle","rb")
MultinomialNB_classifier = pickle.load(save_classifier_f)
save_classifier_f.close()

save_classifier_f = open("pickled_files/BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(save_classifier_f)
save_classifier_f.close()

classifiers = Classification((classifier, MultinomialNB_classifier, BernoulliNB_classifier))

def classification(email):
    return 'is classified as ' + classifiers.classify(feat(email))
