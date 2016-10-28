from gensim import corpora, models, similarities

corpus = corpora.BleiCorpus('./ap/ap.dat', './ap/vocab.txt')

model = models.ldamodel.LdaModel(corpus,num_topics=100,id2word=corpus.id2word)

topics = [model[c] for c in corpus]
##print topics[0]
##[(10, 0.14682159088179139), (16, 0.19212215763578799), (39, 0.013258275624315205), (42, 0.013082145497579998),...]

import numpy as np

dense = np.zeros( (len(topics), 100), float)
##[[ 0.  0.  0. ...,  0.  0.  0.]
## [ 0.  0.  0. ...,  0.  0.  0.]
## [ 0.  0.  0. ...,  0.  0.  0.]....]

for ti,t in enumerate(topics):
     #ti=0,1,2,3.....topic index
     ##t=[(3, 0.053022682448752002), (8, 0.15211882754318831),....] topics
     for tj,v in t:
          dense[ti,tj] = v

from scipy.spatial import distance

pairwise = distance.squareform(distance.pdist(dense))
##Distance of one topic from every other topic
##[[ 0.          0.57654475  0.40657759 ...,  0.59460289  0.69877498
##   0.62654742]
## [ 0.57654475  0.          0.56972099 ...,  0.60785014  0.67146717
##   0.628882  ]....]


largest = pairwise.max()
##1.40406740156

for ti in range(len(topics)):
    pairwise[ti,ti] = largest+1

def closest_to(doc_id):
     ##pairwise[doc_id] = [ 0.58557285  2.40461654  0.75725394 ...,  0.62724108  0.61001829 0.69120095]
     return pairwise[doc_id].argmin()

print(closest_to(7))
##560
##print(topics[closest_to(1)])
