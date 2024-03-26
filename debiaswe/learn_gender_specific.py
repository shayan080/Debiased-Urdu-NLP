from __future__ import print_function, division
import sys
import argparse
from we import WordEmbedding  # Ensure this module is adapted for Urdu
from sklearn.svm import LinearSVC
import json
import numpy as np

# Setup argparse
parser = argparse.ArgumentParser()
parser.add_argument("embedding_filename", help="The name of the embedding file")
parser.add_argument("NUM_TRAINING", type=int, help="Number of training samples")
parser.add_argument("GENDER_SPECIFIC_SEED_WORDS", help="File containing gender-specific seed words in JSON format")
parser.add_argument("outfile", help="Output file for learned gender-specific words")

args = parser.parse_args()

# Assign arguments to variables
embedding_filename = args.embedding_filename
NUM_TRAINING = args.NUM_TRAINING
GENDER_SPECIFIC_SEED_WORDS = args.GENDER_SPECIFIC_SEED_WORDS
OUTFILE = args.outfile

# Load gender-specific seed words
with open(GENDER_SPECIFIC_SEED_WORDS, "r", encoding='utf-8') as f:
    gender_seed = json.load(f)

# Load embeddings
print("Loading embedding...")
E = WordEmbedding(embedding_filename)  # Ensure Urdu embeddings are correctly loaded

# Display stats
print("Embedding has {} words.".format(len(E.words)))
print("{} seed words from '{}' out of which {} are in the embedding.".format(
    len(gender_seed), GENDER_SPECIFIC_SEED_WORDS, len([w for w in gender_seed if w in E.words]))
)

# Create training set
gender_seed = set(w for w in E.words if w in gender_seed)
labeled_train = [(i, 1 if w in gender_seed else 0) for i, w in enumerate(E.words) if (i < NUM_TRAINING or w in gender_seed)]
train_indices, train_labels = zip(*labeled_train)

# Prepare training data
X = np.array([E.vecs[i] for i in train_indices])
y = np.array(train_labels)

# Train classifier
clf = LinearSVC(C=1.0, tol=0.0001, dual=False, class_weight='balanced')
clf.fit(X, y)

# Evaluate and print classifier performance
weights = 1.0 / len(y)  # Update based on training distribution if needed
score = sum((clf.predict(X) == y) * weights)
print("Classifier error rate:", 1 - score, "| Proportion of gender-specific words in training:", sum(y) * 1.0 / len(y))

# Determine gender-specific words
is_gender_specific = (E.vecs.dot(clf.coef_.T) > -clf.intercept_).flatten()
full_gender_specific = list(set([w for label, w in zip(is_gender_specific, E.words) if label]).union(gender_seed))
full_gender_specific.sort(key=lambda w: E.index[w])

# Save learned gender-specific words
try:
    with open(OUTFILE, "w", encoding='utf-8') as f:
        json.dump(full_gender_specific, f)
    print("Saved gender-specific words to", OUTFILE)
except Exception as e:
    print("Error occurred while saving gender-specific words:", str(e))
