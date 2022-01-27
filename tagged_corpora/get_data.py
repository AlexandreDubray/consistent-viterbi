from nltk.corpus import masc_tagged, brown, treebank, mac_morpho, conll2000
import os
import pickle
import nltk
import random

datasets = ["masc_tagged", "brown", "treebank", "conll2000"]
nltk.download(datasets)
nltk.download('universal_tagset')

d = [
    (masc_tagged, 'masc'),
    (brown, 'brown'),
    (treebank, 'treebank'),
    (conll2000, 'conll2000')
]

script_dir = os.path.dirname(os.path.realpath(__file__))

def safe_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def dump_matrix(filename, matrix):
    with open(filename, 'w') as fout:
        fout.write('\n'.join([','.join(["{:.5f}".format(x) for x in row]) for row in matrix]))

def dump_vector(filename, vector):
    with open(filename, 'w') as fout:
        fout.write(','.join(["{:.5f}".format(x) for x in vector]))

def process_corpus(corpus, name, nb_constraint):
    sentences = [x for x in corpus.tagged_sents(tagset='universal')]
    print(len(sentences))
    map_word_int = {}
    next_wid = 0
    map_pos_int = {}
    next_pid = 0

    print('mapping sentences')
    for sentence in sentences:
        for i, (word, pos) in enumerate(sentence):
            if word not in map_word_int:
                map_word_int[word] = next_wid
                next_wid += 1
            if pos not in map_pos_int:
                map_pos_int[pos] = next_pid
                next_pid += 1
            pid = map_pos_int[pos]
            wid = map_word_int[word]
            sentence[i] = (wid, pid)

    directory = os.path.join(script_dir, name + f'_{nb_constraint}')
    safe_mkdir(directory)

    states = [x for x in range(next_pid)]
    random.shuffle(states)
    test_states = set(states[:nb_constraint])

    print('Dumping sentences and tags')
    fsent = open(os.path.join(directory, 'sequences'), 'w')
    ftags = open(os.path.join(directory, 'tags'), 'w')
    ftest_tags = open(os.path.join(directory, 'test_tags'), 'w')
    for sid, sentence in enumerate(sentences):
        fsent.write('\n'.join([f'{sid} {x}' for x, _ in sentence]) + '\n')
        ftags.write('\n'.join([f'{sid} {y}' for _, y in sentence]) + '\n')
        ftest_tags.write(' '.join([f'{sid} {y if y in test_states else -1}' for _, y in sentence]) + '\n')
    fsent.close()
    ftags.close()
    ftest_tags.close()

    with open(os.path.join(directory, 'stats'), 'w') as f:
        f.write(f'nstates {next_pid}\n')
        f.write(f'nobs {next_wid}\n')


for dataset, name in d:
    print(f"{name}...")
    for i in range(2, 8):
        process_corpus(dataset, name, i)
