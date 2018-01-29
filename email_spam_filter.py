import nltk
import scipy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
import random
import io, glob

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

all_words = nltk.FreqDist(all_words)
common_words = list(all_words.keys())

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
random.shuffle(featuresets)

training_set = featuresets[:13500] #40% of featuresets
testing_set = featuresets[13500:] #60% of featuresets

classifier = nltk.NaiveBayesClassifier.train(training_set)

print "Original Naive Bayes Algo accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100
classifier.show_most_informative_features(15)

MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print "MNB_classifier accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100

BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print "BernoulliNB_classifier accuracy percent:", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print "LogisticRegression_classifier accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier,testing_set))*100

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print "SGDClassifier_classifier accuracy percent:", (nltk.classify.accuracy(SGDClassifier_classifier, testing_set))*100

SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print "SVC_classifier accuracy percent:", (nltk.classify.accuracy(SVC_classifier, testing_set))*100

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print "LinearSVC_classifier accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print "NuSVC_classifier accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100
