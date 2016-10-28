from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
     def build_analyzer(self):
          analyzer = super(StemmedCountVectorizer, self).build_analyzer()
          return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedCountVectorizer(min_df=1, stop_words='english')

posts=["This is a toy post about machine learning. Actually, it contains not much interesting stuff.","Imaging databases provide storage capabilities.","Most imaging databases save images permanently.","Imaging databases store data."
,"Imaging databases store data. Imaging databases store data. Imaging databases store data."]

X_train = vectorizer.fit_transform(posts)
#  (0, 23)	1
#  (0, 9)	1
#  (0, 24)	1
#  .
#  .

num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples,num_features))
##samples: 5, #features: 25

print(vectorizer.get_feature_names())
#[u'about', u'actually',...]

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

print(new_post_vec)
#  (0, 5)	1
#  (0, 7)	1

print(new_post_vec.toarray())
#[[0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]]

import scipy as sp
def dist_raw(v1, v2):
     delta = v1-v2
     ##       (0, 7)	-1
     ##       (0, 5)	-1
     ##       (0, 11)	1
     ##       (0, 12)	1
     ##       (0, 0)	1
     ##       (0, 17)	1
     ##       (0, 24)	1
     ##       (0, 9)	1
     ##       (0, 23)	1
     #delta.toarray():[[ 1  0  0  0  0 -1  0 -1  0  1  0  1  1  0  0  0  0  1  0  0  0  0  0  1 1]]
     return sp.linalg.norm(delta.toarray())

def dist_norm(v1, v2):
     v1_normalized = v1/sp.linalg.norm(v1.toarray())
     v2_normalized = v2/sp.linalg.norm(v2.toarray())
     delta = v1_normalized - v2_normalized
     return sp.linalg.norm(delta.toarray())

import sys
best_doc = None
best_dist = sys.maxint
best_i = None
for i in range(0, num_samples):
     post = posts[i]

     post_vec = X_train.getrow(i)
     ##     (0, 23)	1
     ##     (0, 9)	1
     ##     (0, 24)	1
     ##     (0, 17)	1
     ##     (0, 0)	1
     ##     (0, 12)	1
     ##     (0, 11)	1
     
     d = dist_norm(post_vec, new_post_vec)
     print "=== Post %i with dist=%.2f: %s"%(i, d, post)
     if d<best_dist:
          best_dist = d
          best_i = i
     print("Best post is %i with dist=%.2f"%(best_i, best_dist))

