''' Formats words, supertags, and parts of speech for input to the MICA parser'''
DATA_DIR = 'data/'
PREFIXES = ['train', 'test']
TYPES = ['ref', 'para']

for pre in PREFIXES:
    for typ in TYPES:
        stag_seqs = []
        with open(DATA_DIR + pre + '-' + typ + '-supertags.txt', 'r') as f:
            for line in f:
                stags = line.split()
                stag_seqs.append(stags)

        word_seqs = []
        with open(DATA_DIR + pre + '-' + typ + '-output.txt', 'r') as f:
            for line in f:
                words = line.split()
                word_seqs.append(words)

        pos_seqs = []
        with open(DATA_DIR + pre + '-' + typ + '-tags.txt', 'r') as f:
            for line in f:
                pos = line.split()
                pos_seqs.append(pos)



        with open(DATA_DIR + pre + '-' + typ + '-mica-input.txt', 'w') as f: 
            count = 0
            for i in range(len(stag_seqs)):
                stags = stag_seqs[i]
                words = word_seqs[i]
                pos = pos_seqs[i]
                if len(stags) != len(words) or len(stags) != len(pos) or len(words) != len(pos):
                    count +=1
                f.write("##SDAG BEGIN /* sent_id={} length={} trans_nb=15 max_lexical_ambiguity=1 */\n".format(i+1, len(stags)))
                for j in range(min([len(stags), len(words), len(pos)])):
                    f.write('"{} {}" ({} [|1.0000|] )\n'.format(words[j], pos[j], stags[j]))
                f.write('##SDAG END\n')
            print(count)