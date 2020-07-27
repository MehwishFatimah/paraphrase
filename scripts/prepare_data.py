''' Generates part of speech tags for a set of sentences '''
import nltk
import random

# location of paired data 
DATA_PATH = 'data/pairs.txt'
# how much data to set aside for testing
TEST_PCT = 0.2

def tag_sents(sents, sent_path, tag_path):
    with open(sent_path, 'w') as sentf:
        with open(tag_path, 'w') as tagf:
            for sent in sents:
                tokens = nltk.word_tokenize(sent)
                word_tags = nltk.pos_tag(tokens)
                words, tags = zip(*word_tags)
                tagf.write(" ".join(tags) + '\n')
                sentf.write(" ".join(words) + '\n')

if __name__ == "__main__":
    pairs = []
    with open(DATA_PATH, 'r') as pair_file:
        for line in pair_file:
            ref, para = line.strip().split('\t')
            pairs.append((ref, para))

    random.shuffle(pairs)

    break_pt = int(TEST_PCT * len(pairs))
    test_pairs = pairs[:break_pt]
    train_pairs = pairs[break_pt:]

    train_ref, train_para = zip(*train_pairs)
    test_ref, test_para = zip(*test_pairs)

    tag_sents(train_ref, 'data/train-ref-words.txt', 'data/train-ref-tags.txt')
    tag_sents(train_para, 'data/train-para-words.txt', 'data/train-para-tags.txt')
    tag_sents(test_ref, 'data/test-ref-words.txt', 'data/test-ref-tags.txt')
    tag_sents(test_para, 'data/test-para-words.txt', 'data/test-para-tags.txt')

