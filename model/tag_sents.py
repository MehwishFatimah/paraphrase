''' Generates part of speech tags for a set of sentences '''
import nltk

if __name__ == "__main__":
    filename = 'new-data-bidir-lin-100/'

    with open(filename + "test-bidir-100.txt", 'r') as f:
        with open(filename + 'test-bidir-100-tags.txt', 'w') as t:
            for line in f:
                tokens = nltk.word_tokenize(line)
                word_tags = nltk.pos_tag(tokens)
                words, tags = zip(*word_tags)
                t.write(" ".join(tags) + '\n')

