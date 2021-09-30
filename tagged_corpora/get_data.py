from nltk.corpus import masc_tagged, brown, treebank, mac_morpho
import os
import pickle
import nltk

nltk.download(["masc_tagged", "brown", "treebank", "mac_morpho"])

def dump_matrix(filename, matrix):
    with open(filename, 'w') as fout:
        fout.write('\n'.join([' '.join([str(x) for x in row]) for row in matrix]))

def dump_vector(filename, vector):
    with open(filename, 'w') as fout:
        fout.write(' '.join([str(x) for x in vector]))

def process_corpus(corpus, name):
    sentences = [x for x in corpus.tagged_sents()]
    print(len(sentences))
    map_word_int = {}
    map_word_int_r = {}
    next_wid = 0
    map_pos_int = {}
    map_pos_int_r = {}
    next_pid = 0

    print('mapping sentences')
    for sentence in sentences:
        for i, (word, pos) in enumerate(sentence):
            if word not in map_word_int:
                map_word_int[word] = next_wid
                map_word_int_r[next_wid] = word
                next_wid += 1
            if pos not in map_pos_int:
                map_pos_int[pos] = next_pid
                map_pos_int_r[next_pid] = pos
                next_pid += 1
            sentence[i] = (map_word_int[word], map_pos_int[pos])
    pi = [0 for _ in range(next_pid)]
    A = [[0 for _ in range(next_pid)] for _ in range(next_pid)]
    b = [[0 for _ in range(next_wid)] for _ in range(next_pid)]
    seen_states = [0 for _ in range(next_pid)]

    print('computing HMM structure')
    for sentence in sentences:
        pi[sentence[0][1]] += 1
        for i in range(len(sentence)):
            (word_i, pos_i) = sentence[i]
            if i < len(sentence) - 1:
                (_, pos_j) = sentence[i+1]
                A[pos_i][pos_j] += 1

            b[pos_i][word_i] += 1
            seen_states[pos_i] += 1

    pi = [x/len(sentences) for x in pi]
    A = [[x/seen_states[i] for x in row] for i, row in enumerate(A)]
    b = [[x/seen_states[i] for x in row] for i, row in enumerate(b)]

    try:
        os.mkdir(name)
    except FileExistsError:
        pass

    print('Dumping HMM structure...')
    dump_matrix(os.path.join(name, 'A'), A)
    dump_matrix(os.path.join(name, 'b'), b)
    dump_vector(os.path.join(name, 'pi'), pi)

    print('Dumping sentences and tags')
    fsent = open(os.path.join(name, 'sentences'), 'w')
    ftags = open(os.path.join(name, 'tags'), 'w')
    for sentence in sentences:
        fsent.write(' '.join([str(x) for x, _ in sentence]) + '\n')
        ftags.write(' '.join([str(y) for _, y in sentence]) + '\n')
    fsent.close()
    ftags.close()
    print('Dumping maps')
    with open(os.path.join(name, 'map_word.pkl'), 'wb') as f:
        pickle.dump(map_word_int_r, f)
    with open(os.path.join(name, 'map_pos.pkl'), 'wb') as f:
        pickle.dump(map_pos_int_r, f)

print('masc...')
process_corpus(masc_tagged, 'masc')
print('brown...')
process_corpus(brown, 'brown')
print('treebank...')
process_corpus(treebank, 'treebank')
print('mac_morpho...')
process_corpus(mac_morpho, 'mac_morpho')
