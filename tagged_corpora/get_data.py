from nltk.corpus import masc_tagged, brown, treebank, mac_morpho
import os
import pickle
import nltk

nltk.download(["masc_tagged", "brown", "treebank", "mac_morpho"])

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

    consistency_constraints = [set() for _ in range(next_pid)]

    print('computing HMM structure')
    for sid, sentence in enumerate(sentences):
        pi[sentence[0][1]] += 1
        for i in range(len(sentence)):
            (word_i, pos_i) = sentence[i]
            consistency_constraints[pos_i].add((sid, i))
            if i < len(sentence) - 1:
                (_, pos_j) = sentence[i+1]
                A[pos_i][pos_j] += 1

            b[pos_i][word_i] += 1
            seen_states[pos_i] += 1

    pi = [x/len(sentences) for x in pi]
    A = [[x/seen_states[i] for x in row] for i, row in enumerate(A)]
    b = [[x/seen_states[i] for x in row] for i, row in enumerate(b)]

    safe_mkdir(os.path.join(script_dir, name))

    print('Dumping HMM structure...')
    dump_matrix(os.path.join(script_dir, name, 'A'), A)
    dump_matrix(os.path.join(script_dir, name, 'b'), b)
    dump_vector(os.path.join(script_dir, name, 'pi'), pi)

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
    safe_mkdir(os.path.join(script_dir, name, 'configs'))

    for method in ['viterbi', 'global_opti']:
        output_path = os.path.join(script_dir, name, 'output_{method}')
        safe_mkdir(output_path)
        for prop_int in range(0, 101, 5):
            prop = round(prop_int / 100, 2)
            with open(os.path.join(script_dir, name, 'configs', f'config_{method}'), 'w') as f:
                f.write(f'method={method}\n')
                f.write(f'hmm_path={os.path.join(script_dir, name)}\n')
                f.write(f'input_path={os.path.join(script_dir, name)}\n')
                f.write(f'output_path={output_path}')
                f.write(f'nstates={len(A)}\n')
                f.write(f'nobs={len(b[0])}\n')
                f.write(f'prop={prop}\n')


print('masc...')
process_corpus(masc_tagged, 'masc')
print('brown...')
process_corpus(brown, 'brown')
print('treebank...')
process_corpus(treebank, 'treebank')
print('mac_morpho...')
process_corpus(mac_morpho, 'mac_morpho')
