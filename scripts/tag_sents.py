import nltk

sent_path = 'data/test-bidir-lin-100-output.txt'
tag_path = 'data/test-bidir-lin-100-tags.txt'

with open(sent_path, 'r') as sentf:
    with open(tag_path, 'w') as tagf:
        for sent in sentf:
            tokens = nltk.word_tokenize(sent)
            word_tags = nltk.pos_tag(tokens)
            words, tags = zip(*word_tags)
            tagf.write(" ".join(tags) + '\n')