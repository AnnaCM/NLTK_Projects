import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import pickle, scipy
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB

import random, io, glob

all_emails = []
for f in glob.glob("Enron-Spam\*\ham\*.txt"):
    with io.open(f,'r') as file_name:
        all_emails.append((file_name.read(), 'non_spam'))

for f in glob.glob("Enron-Spam\*\spam\*.txt"):
    with io.open(f,'r', encoding='latin-1') as file_name:
        all_emails.append((file_name.read(), 'spam'))

for e,c in all_emails:
    words = word_tokenize(e)
tagged = nltk.pos_tag(words)
type_of_words = ['NN','NNS','NNP','NNPS','JJ','JJR','JJS','MD','LS','RB','RBR','RBS','VB','VBD','VBG','VBN','VBP','VBZ']
eng_stop_words = set(stopwords.words('english'))
all_words = []
for w,t in tagged:
    if w not in eng_stop_words and t in type_of_words:
        all_words.append(w.lower())

save_emails = open("pickled_files/all_emails.pickle","wb")
pickle.dump(all_emails, save_emails)
save_emails.close()

all_words = nltk.FreqDist(all_words)
common_words = list(all_words.keys())

save_common_words = open("pickled_files/common_words.pickle","wb")
pickle.dump(common_words, save_common_words)
save_common_words.close()

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

featuresets = [(feat(email),category) for (email,category) in all_emails]

save_featuresets = open("pickled_files/featuresets.pickle","wb")
pickle.dump(featuresets, save_featuresets)
save_featuresets.close()

random.shuffle(featuresets)
training_set = featuresets[:27000] #80% of featuresets
testing_set = featuresets[27000:] #20% of featuresets

classifier = nltk.NaiveBayesClassifier.train(training_set)
save_classifier = open("pickled_files/naivebayes.pickle","wb")
pickle.dump(classifier, save_classifier)
save_classifier.close()

MultinomialNB_classifier = SklearnClassifier(MultinomialNB())
MultinomialNB_classifier.train(training_set)
save_classifier = open("pickled_files/MultinomialNB_classifier.pickle","wb")
pickle.dump(MultinomialNB_classifier, save_classifier)
save_classifier.close()

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
save_classifier = open("pickled_files/BernoulliNB_classifier.pickle","wb")
pickle.dump(BernoulliNB_classifier, save_classifier)
save_classifier.close()
