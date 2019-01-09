from nltk.corpus import movie_reviews
from featx import label_feats_from_corpus, split_label_feats
from nltk.classify import DecisionTreeClassifier
from nltk.classify.util import accuracy
def dt_classifier():
   cat=movie_reviews.categories()
   lfeats = label_feats_from_corpus(movie_reviews)
   k1=lfeats.keys()
   train_feats, test_feats = split_label_feats(lfeats)
   dt_classifier = DecisionTreeClassifier.train(train_feats,binary=True, entropy_cutoff=0.8, depth_cutoff=5, support_cutoff=30)
   acc=accuracy(dt_classifier, test_feats)
   print "The Accuracy Is:",acc
   
