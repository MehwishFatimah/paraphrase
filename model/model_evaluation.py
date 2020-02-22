DATA_DIR = '../data/artificial-data/set-2/test/'

def read_seqs(filename):
    seq = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(' ')
            seq.append(line)
    return seq

def item_for_item_accuracy(gold, output):
    correct_items = 0
    for gold_item, output_item in zip(gold, output):
        if gold_item == output_item:
            correct_items += 1

    acc = correct_items * 1.0 / len(gold)
    return acc

def avg_item_for_item_accuracy(gold_seqs, output_seqs):
    total_acc = 0
    for gold, output in zip(gold_seqs, output_seqs):
        total_acc += item_for_item_accuracy(gold, output)
    
    avg_acc = total_acc / len(gold_seqs)
    return avg_acc

def word_for_supertag_accuracy(gold_words, output_words, gold_stags, output_stags):
    correct_words = 0
    correct_supertags = 0
    for gold_word, output_word, gold_stag, output_stag in zip(gold_words, output_words, gold_stags, output_stags):
        if gold_stag == output_stag:
            correct_supertags += 1
            if gold_word == output_word:
                correct_words += 1
    if correct_supertags == 0:
        return None
    acc = correct_words * 1.0 / correct_supertags
    return acc

def avg_word_for_supertag_accuracy(gold_sents, output_sents, gold_supertags, output_supertags):
    total_acc = 0
    total_sents = 0
    for gold_sent, output_sent, gold_stags, output_stags in zip(gold_sents, output_sents, gold_supertags, output_supertags):
        acc = word_for_supertag_accuracy(gold_sent, output_sent, gold_stags, output_stags)
        if acc is not None:
            total_acc += acc
            total_sents += 1

    avg_acc = total_acc / total_sents
    return avg_acc

ref_sents = read_seqs(DATA_DIR + 'test-ref-words.txt')
para_sents = read_seqs(DATA_DIR + 'test-para-words.txt')
output_sents = read_seqs(DATA_DIR + 'test-output.txt')

para_supertags = read_seqs(DATA_DIR + 'test-para-supertags.txt')
output_supertags = read_seqs(DATA_DIR + 'test-output-supertags.txt')

print('Average word-for-word accuracy: ', avg_item_for_item_accuracy(para_sents, output_sents))
print('Average supertag-for-supertag accuracy: ', avg_item_for_item_accuracy(para_supertags, output_supertags))
print('Of correct supertags, average word accuracy: ', avg_word_for_supertag_accuracy(para_sents, output_sents, para_supertags, output_supertags))
