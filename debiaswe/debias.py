from __future__ import print_function, division
import we
import json
import numpy as np
import argparse
import sys
if sys.version_info[0] < 3:
    import io
    open = io.open


def debias(E, gender_specific_words, definitional, equalize):
 
    gender_direction = we.doPCA(definitional, E).components_[0]
    specific_set = set(gender_specific_words)
    for i, w in enumerate(E.words):
        if w not in specific_set:
            E.vecs[i] = we.drop(E.vecs[i], gender_direction)
    E.normalize()
    
   
    # as Urdu does not use upper and lower cases like Latin script.
    candidates = {(e1, e2) for e1, e2 in equalize}
    
    for (a, b) in candidates:
        if (a in E.index and b in E.index):
            mid_point = (E.v(a) + E.v(b)) / 2
            y = we.drop(mid_point, gender_direction)
            z = np.sqrt(1 - np.linalg.norm(y)**2)
            direction = (E.v(a) - E.v(b)).dot(gender_direction)
            z = -z if direction < 0 else z
            E.vecs[E.index[a]], E.vecs[E.index[b]] = z * gender_direction + y, -z * gender_direction + y
    E.normalize()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding_filename", help="The name of the embedding file.")
    parser.add_argument("definitional_filename", help="JSON file of definitional pairs.")
    parser.add_argument("gendered_words_filename", help="File containing words not to neutralize (JSON).")
    parser.add_argument("equalize_filename", help="JSON file of word pairs to equalize.")
    parser.add_argument("debiased_filename", help="Filename for the debiased embeddings.")

    args = parser.parse_args()

    # Load definitional pairs, gender-specific words, and equalize pairs from JSON files
    with open(args.definitional_filename, "r", encoding='utf-8') as f:
        definitional_pairs = json.load(f)

    with open(args.gendered_words_filename, "r", encoding='utf-8') as f:
        gender_specific_words = json.load(f)

    with open(args.equalize_filename, "r", encoding='utf-8') as f:
        equalize_pairs = json.load(f)

    # Load embeddings
    E = we.WordEmbedding(args.embedding_filename)

    # Perform debiasing
    debias(E, gender_specific_words, definitional_pairs, equalize_pairs)

    # Save debiased embeddings
    if args.debiased_filename.endswith(".bin"):
        E.save_w2v(args.debiased_filename, binary=True)
    else:
        E.save(args.debiased_filename)

    print("Debiasing completed and saved to", args.debiased_filename)
