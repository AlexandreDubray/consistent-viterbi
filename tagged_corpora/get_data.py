from nltk.corpus import masc_tagged, brown, treebank, mac_morpho, conll2000
import os
import pickle
import nltk

datasets = ["masc_tagged", "brown", "treebank", "mac_morpho", "conll2000"]
nltk.download(datasets)

d = [
    (masc_tagged, 'masc'),
    (brown, 'brown'),
    (treebank, 'treebank'),
    (mac_morpho, 'mac_morpho'),
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

def process_corpus(corpus, name):
    sentences = [x for x in corpus.tagged_sents()]
    print(len(sentences))
    map_word_int = {}
    map_word_int_r = {}
    next_wid = 0
    map_pos_int = {}
    map_pos_int_r = {}
    next_pid = 0

    consistency_constraints = []

    print('mapping sentences')
    for seq_id, sentence in enumerate(sentences):
        for i, (word, pos) in enumerate(sentence):
            if word not in map_word_int:
                map_word_int[word] = next_wid
                map_word_int_r[next_wid] = word
                next_wid += 1
            if pos not in map_pos_int:
                map_pos_int[pos] = next_pid
                map_pos_int_r[next_pid] = pos
                next_pid += 1
                consistency_constraints.append(set())
            pid = map_pos_int[pos]
            wid = map_word_int[word]
            consistency_constraints[pid].add((seq_id, i))
            sentence[i] = (wid, pid)

    safe_mkdir(os.path.join(script_dir, name))

    print('Dumping sentences and tags')
    fsent = open(os.path.join(script_dir, name, 'sequences'), 'w')
    ftags = open(os.path.join(script_dir, name, 'tags'), 'w')
    for sentence in sentences:
        fsent.write(' '.join([str(x) for x, _ in sentence]) + '\n')
        ftags.write(' '.join([str(y) for _, y in sentence]) + '\n')
    fsent.close()
    ftags.close()

    print("Dumping consistency constraints")
    with open(os.path.join(script_dir, name, "constraints"), "w") as f:
        f.write('\n\n'.join([ '\n'.join([f'{seq_id} {t}' for seq_id, t in component]) for component in consistency_constraints]))
    print('Dumping maps')
    with open(os.path.join(script_dir, name, 'map_word.pkl'), 'wb') as f:
        pickle.dump(map_word_int_r, f)
    with open(os.path.join(script_dir, name, 'map_pos.pkl'), 'wb') as f:
        pickle.dump(map_pos_int_r, f)

    print("Generating configs")
    #for method in ['viterbi', 'global_opti', 'dp']:
    for method in ['ilp']:
        output_path = os.path.join(script_dir, name, f'output_{method}')
        config_path = os.path.join(script_dir, name, f'configs_{method}')
        safe_mkdir(output_path)
        safe_mkdir(config_path)
        for prop_int in range(0, 101, 5):
            prop = round(prop_int / 100, 2)
            with open(os.path.join(config_path, f'config_{prop}'), 'w') as f:
                f.write(f'method={method}\n')
                f.write(f'input_path={os.path.join(script_dir, name)}\n')
                f.write(f'output_path={output_path}\n')
                f.write(f'nstates={next_pid}\n')
                f.write(f'nobs={next_wid}\n')
                f.write(f'prop={prop}\n')


for dataset, name in d:
    print(f"{name}...")
    process_corpus(dataset, name)
