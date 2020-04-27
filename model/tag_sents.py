import nltk

if __name__ == "__main__":
    filename = '../data/artificial-data/set-2/test/test'

    with open(filename + "-opennmt-output.txt", 'r') as f:
        with open(filename + '-opennmt-tags.txt', 'w') as t:
            for line in f:
                tokens = nltk.word_tokenize(line)
                word_tags = nltk.pos_tag(tokens)
                words, tags = zip(*word_tags)
                t.write(" ".join(tags) + '\n')

