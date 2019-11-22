stag_seqs = []
with open('opennmt-pred.txt', 'r') as f:
    for line in f:
        stags = line.split()
        stag_seqs.append(stags)

word_seqs = []
with open('test-para-words.txt', 'r') as f:
    for line in f:
        words = line.split()
        word_seqs.append(words)

pos_seqs = []
with open('test-para-tags.txt', 'r') as f:
    for line in f:
        pos = line.split()
        pos_seqs.append(pos)

assert(len(stag_seqs) == len(word_seqs))
assert(len(stag_seqs) == len(pos_seqs))


with open('mica-parser-input-opennmt.txt', 'w') as f: 
    for i in range(len(stag_seqs)):
        stags = stag_seqs[i]
        words = word_seqs[i]
        pos = pos_seqs[i]

        f.write("##SDAG BEGIN /* sent_id={} length={} trans_nb=15 max_lexical_ambiguity=1 */\n".format(i+1, len(stags)))
        for j in range(len(stags)):
            f.write('"{} {}" ({} [|1.0000|] )\n'.format("UNK", "--UNK--", stags[j]))
        f.write('##SDAG END\n')