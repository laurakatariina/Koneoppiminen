import random
import numpy as np

vocabulary_file = 'C:/Users/Laura/Documents/Yliopisto/dataml100/word_embeddings.txt'

def get_nearest_neighbors(vector, n):
    # Calculate Euclidean distances between the input vector and each vector in the matrix W
    distances = np.linalg.norm(W - vector, axis=1)
    
    # Get the indices of the n closest vectors
    nearest_indices = np.argpartition(distances, n)[:n]
    
    # Sort distances of the nearest neighbors
    sorted_indices = np.argsort(distances[nearest_indices])
    sorted_nearest_indices = nearest_indices[sorted_indices]
    
    return sorted_nearest_indices, distances

# Read words
print('Read words...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    words = [x.rstrip().split(' ')[0] for x in f.readlines()]

# Read word vectors
print('Read word vectors...')
with open(vocabulary_file, 'r', encoding="utf8") as f:
    vectors = {}
    for line in f:
        vals = line.rstrip().split(' ')
        vectors[vals[0]] = [float(x) for x in vals[1:]]

vocab_size = len(words)
vocab = {w: idx for idx, w in enumerate(words)}
ivocab = {idx: w for idx, w in enumerate(words)}

# Vocabulary and inverse vocabulary (dict objects)
print('Vocabulary size')
print(len(vocab))
print(vocab['man'])  # Example
print(len(ivocab))
print(ivocab[10])   # Example

# W contains vectors for the vocabulary
print('Vocabulary word vectors')
vector_dim = len(vectors[ivocab[0]])
W = np.zeros((vocab_size, vector_dim))
for word, v in vectors.items():
    if word == '<unk>':
        continue
    W[vocab[word], :] = v
print(W.shape)

def analogy(a, b, c):
    if a not in vocab or b not in vocab or c not in vocab:
        raise ValueError("One or more words are not in the vocabulary.")
    
    # Compute the analogy vector
    analogy_vector = W[vocab[c]] + (W[vocab[b]] - W[vocab[a]])
    
    # Find the nearest neighbors to the analogy vector
    nearest_indices, distances = get_nearest_neighbors(analogy_vector, 2)
    
    # Collect the best matches
    best_matches = [(ivocab[idx], distances[idx]) for idx in nearest_indices]
    
    return best_matches

# Main loop for analogy
while True:
    input_term = input("\nEnter the first word (EXIT to break): ")
    if input_term == 'EXIT':
        break
    input_term2 = input("Enter the second word: ")
    input_term3 = input("Enter the third word: ")

    try:
        matches = analogy(input_term, input_term2, input_term3)
        print("\n                               Word       Distance\n")
        print("---------------------------------------------------------\n")
        
        for word, distance in matches:
            print("%35s\t\t%f" % (word, distance))
    
    except ValueError as e:
        print(f"Error: {e}")