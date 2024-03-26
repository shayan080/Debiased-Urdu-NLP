from __future__ import print_function, division
import re
import sys
from typing import Counter
import numpy as np
import gensim.models
from gensim.models import word2vec
from gensim.models import KeyedVectors
import scipy.sparse
from sklearn.decomposition import PCA
if sys.version_info[0] < 3:
    import io
    open = io.open
else:
    unicode = str



def dedup(seq):
    seen = set()
    return [x for x in seq if not (x in seen or seen.add(x))]

def safe_word(w):
    # Adjusted to include Urdu characters
    # Urdu characters range: U+0600 to U+06FF, U+0750 to U+077F, plus some additions
    urdu_char_range = r'[\u0600-\u06FF\u0750-\u077F]'
    return bool(re.match(urdu_char_range + '+$', w)) and len(w) < 20


def to_utf8(text, errors='strict', encoding='utf8'):
    if isinstance(text, unicode):
        return text.encode('utf8')
    # do bytestring -> unicode -> utf8 full circle, to ensure valid utf8
    return unicode(text, encoding, errors=errors).encode('utf8')



class WordEmbedding:
    def __init__(self, fname):
        self.thresh = None
        self.max_words = None
        self.desc = fname
        print("*** Reading data from " + fname)
        try:
            if fname.endswith(".bin"):
                # Attempt to load model using Gensim
                model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
                words = model.index_to_key  # Gensim 4.0.0+ uses index_to_key for word list
                vecs = model.vectors  # Use .vectors to access the numpy array of vectors directly
            else:
                vecs, words = self.load_text_format(fname)
            
            self.vecs = np.array(vecs, dtype='float32')
            print(self.vecs.shape)
            self.words = words
            self.reindex()
            norms = np.linalg.norm(self.vecs, axis=1)
            if max(norms)-min(norms) > 0.0001:
                self.normalize()
        except Exception as e:
            print("Warning: There was an issue loading the embeddings.")
            print("Error details:", e)

    
    def load_text_format(self, fname):
        vecs = []
        words = []
        with open(fname, "r", encoding='utf8') as f:
            for line in f:
                s = line.split()
                if len(s) < 10:  # Skip incorrect lines
                    continue
                word, vector = s[0], list(map(float, s[1:]))
                words.append(word)
                vecs.append(vector)
        return vecs, words



    def reindex(self):
        self.index = {w: i for i, w in enumerate(self.words)}
        self.n, self.d = self.vecs.shape
        assert self.n == len(self.words) == len(self.index)
        self._neighbors = None
        print(self.n, "words of dimension", self.d, ":", ", ".join(self.words[:4] + ["..."] + self.words[-4:]))

    def v(self, word):
        return self.vecs[self.index[word]]

    def diff(self, word1, word2):
        v = self.vecs[self.index[word1]] - self.vecs[self.index[word2]]
        return v/np.linalg.norm(v)

    def normalize(self):
        self.desc += ", normalize"
        self.vecs /= np.linalg.norm(self.vecs, axis=1)[:, np.newaxis]
        self.reindex()

    def shrink(self, numwords):
        self.desc += ", shrink " + str(numwords)
        self.filter_words(lambda w: self.index[w]<numwords)

    def filter_words(self, test):
       
        self.desc += ", filter"
        kept_indices, words = zip(*[[i, w] for i, w in enumerate(self.words) if test(w)])
        self.words = list(words)
        self.vecs = self.vecs[np.array(kept_indices), :]
        self.reindex()

    def save(self, filename):
        with open(filename, "w", encoding='utf-8') as f:
            f.write("\n".join([w + " " + " ".join([str(x) for x in v]) for w, v in zip(self.words, self.vecs)]))
        print("Wrote", len(self.words), "words to", filename)

    def save_w2v(self, filename, binary=True):
        with open(filename, 'wb') as fout:
            fout.write(("%s %s\n" % self.vecs.shape).encode('utf-8'))
            for i, word in enumerate(self.words):
                row = self.vecs[i]
                if binary:
                    fout.write(word.encode('utf-8') + b" " + row.tobytes())
                else:
                    fout.write(("%s %s\n" % (word, ' '.join("%f" % val for val in row))).encode('utf-8'))

    def remove_directions(self, directions, numwords=None):
    
        self.desc += ", removed"
        for direction in directions:
            self.desc += " "
            if isinstance(direction, np.ndarray):
                v = direction / np.linalg.norm(direction)
                self.desc += "vector "
            else:
                w1, w2 = direction
                v = self.diff(w1, w2)
                self.desc += w1 + "-" + w2
            self.vecs = self.vecs - self.vecs.dot(v)[:, np.newaxis] .dot( v[np.newaxis, :])
        self.normalize()
        



    def compute_neighbors_if_necessary(self, thresh, max_words):
        thresh = float(thresh)
        if self._neighbors is not None and self.thresh == thresh and self.max_words == max_words:
            return
        print("Computing neighbors")
        self.thresh = thresh
        self.max_words = max_words
        vecs = self.vecs[:max_words]
        dots = vecs @ vecs.T  # Using @ for matrix multiplication in newer Python versions
        # Creating a sparse matrix where entries less than the threshold are discarded
        dots = scipy.sparse.csr_matrix(dots * (dots >= 1 - thresh / 2))
        rows, cols = dots.nonzero()
        nums = list(Counter(rows).values())
        print("Mean:", np.mean(nums) - 1)
        print("Median:", np.median(nums) - 1)
        # Filter and normalize vectors for non-zero dot product entries
        rows, cols, vecs = zip(*[(i, j, vecs[i] - vecs[j]) for i, j in zip(rows, cols) if i < j])
        self._neighbors = rows, cols, np.array([v / np.linalg.norm(v) for v in vecs])

    def neighbors(self, word, thresh=1):
        if word not in self.index:
            return []
        dots = self.vecs.dot(self.v(word))
        return [self.words[i] for i, dot in enumerate(dots) if dot >= 1 - thresh / 2]

    def more_words_like_these(self, words, topn=50, max_freq=100000):
        vectors = [self.v(w) for w in words if w in self.index]
        if not vectors:
            return []
        v = sum(vectors)
        dots = self.vecs[:max_freq].dot(v)
        thresh = sorted(dots)[-topn]
        words = [w for w, dot in zip(self.words, dots) if dot >= thresh]
        return sorted(words, key=lambda w: self.v(w).dot(v), reverse=True)[:topn]
    
    def best_analogies_dist_thresh(self, v, thresh=1, topn=500, max_words=50000):
      
        vecs, vocab = self.vecs[:max_words], self.words[:max_words]
        self.compute_neighbors_if_necessary(thresh, max_words)
        rows, cols, neighbor_vecs = self._neighbors
        scores = neighbor_vecs.dot(v / np.linalg.norm(v))
        pi = np.argsort(-abs(scores))

        ans = []
        usedL, usedR = set(), set()
        for i in pi:
            if abs(scores[i]) < 0.001:
                break
            row, col = (rows[i], cols[i]) if scores[i] > 0 else (cols[i], rows[i])
            if row in usedL or col in usedR:
                continue
            usedL.add(row)
            usedR.add(col)
            ans.append((vocab[row], vocab[col], abs(scores[i])))
            if len(ans) == topn:
                break

        return ans
    

    

def viz(analogies):
    for i, a in enumerate(analogies):
        # Concatenates the analogy components with proper spacing and alignment
        print(f"{str(i+1).rjust(4)}: {a[0].rjust(20)} -> {a[1].ljust(20)} Similarity: {a[2]:.4f}")





def text_plot_words(xs, ys, words, width=90, height=40, filename=None):
    PADDING = 10  # num chars on left and right in case words spill over
    res = [[' ' for _ in range(width)] for _ in range(height)]

    def rescale(nums):
        a, b = min(nums), max(nums)
        return [(x - a) / (b - a) for x in nums]

    xs, ys = rescale(xs), rescale(ys)
    for (x, y, word) in zip(xs, ys, words):
        i, j = int(x * (width - 1 - PADDING)), int(y * (height - 1))
        if any(row[max(i - 1, 0):min(width, i + len(word) + 1)] != ' ' for row in res[j]):
            continue  # Skip if overlap
        for k, char in enumerate(word):
            if i + k >= width: break
            res[j][i + k] = char

    string = "\n".join("".join(row) for row in res)
    if filename:
        with open(filename, "w", encoding="utf8") as f:
            f.write(string)
        print("Wrote to", filename)
    else:
        print(string)


def doPCA(pairs, embedding, num_components = 10):
    matrix = []
    for a, b in pairs:
        center = (embedding.v(a) + embedding.v(b))/2
        matrix.append(embedding.v(a) - center)
        matrix.append(embedding.v(b) - center)
    matrix = np.array(matrix)
    pca = PCA(n_components = num_components)
    pca.fit(matrix)
    # bar(range(num_components), pca.explained_variance_ratio_)
    return pca


def drop(u, v):
    return u - v * u.dot(v) / v.dot(v)

