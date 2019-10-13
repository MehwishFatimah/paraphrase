import nltk.data
import nltk
# tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# fp = open("test.txt")
# data = fp.read()
# print '\n-----\n'.join(tokenizer.tokenize(data))

if __name__ == "__main__":

    with open('test-para-sents.txt', 'r') as f:
        with open('test-para-words.txt', 'w') as w:
            with open('test-para-tags.txt', 'w') as t:
                for line in f:
                    tokens = nltk.word_tokenize(line)
                    word_tags = nltk.pos_tag(tokens)
                    words, tags = zip(*word_tags)
                    w.write(" ".join(words) + '\n')
                    t.write(" ".join(tags) + '\n')



    # with open('para-nmt-5m-processed.txt', 'r') as f:
    #     with open('para-nmt-5m-processed-ref.txt', 'w') as r:
    #         with open('para-nmt-5m-processed-para.txt', 'w') as p:
    #             for line in f:
    #                 split = line.split('\t')
    #                 r.write(split[0]+'\n')
    #                 p.write(split[1])
                 
    # with open('para-nmt-5m-processed-ref.txt', 'r') as a:
    #     with open('test-ref-sents.txt', 'w') as t:
    #         i = 0
    #         for line in a:
    #             if i < 100000:
    #                 l = tokenizer.tokenize(line)
    #                 t.write(l[0] + '\n')
    #                 i += 1

    # with open('para-nmt-5m-processed-para.txt', 'r') as b:
    #     with open('test-para-sents.txt', 'w') as t:
    #         i = 0
    #         for line in b:
    #             if i < 100000:
    #                 l = tokenizer.tokenize(line)
    #                 t.write(l[0] + '\n')
    #             i += 1

    # split the POS tags from the sentences
    # with open('test-ref-tags.txt', 'r') as f:
    #     with open('test-ref-tagged-sents.txt', 'w') as a:
    #         with open('test-ref-tagged-tags.txt', 'w') as b:
    #             for line in f:
    #                 tokens = line.split()
    #                 words = [token.split('_')[0] for token in tokens]
    #                 sent = ' '.join(words) + '\n'
    #                 a.write(sent)
    #                 pos = [token.split('_')[1] if len(token.split('_')) > 1 else 'UNK' for token in tokens]
    #                 seq = ' '.join(pos) + '\n'
    #                 b.write(seq)

