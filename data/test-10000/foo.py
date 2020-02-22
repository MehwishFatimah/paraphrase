import unicodedata
import re
MAX_LENGTH = 10

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
            len(p[2].split(' ')) < MAX_LENGTH


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def readLangs(lang1, lang2, test=False, reverse=False):
    print("Reading lines...")
    data_dir = '../data/train-100000'
    prefix = 'train'
    if test:
        data_dir = '../data/test-10000'
        prefix = 'test'

    # Read the file and split into lines
    ref_lines = open('{}-ref-words.txt'.format(prefix), encoding='utf-8').\
        read().strip().split('\n')
    
    para_lines = open('{}-para-words.txt'.format(prefix), encoding='utf-8').\
        read().strip().split('\n')

    tags = open('{}-para-supertags.txt'.format(prefix), encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = list(zip([normalizeString(l) for l in ref_lines], [normalizeString(l) for l in para_lines], tags))

    return pairs

pairs = readLangs('ref', 'para', True, False)
print("Read %s sentence pairs" % len(pairs))
pairs = filterPairs(pairs)
print("Trimmed to %s sentence pairs" % len(pairs))


with open('test-ref-filter10.txt', 'w') as f:
    for pair in pairs:
        f.write(pair[0] + '\n')

with open('test-para-filter10.txt', 'w') as f:
    for pair in pairs:
        f.write(pair[1] + '\n')

with open('test-supertag-filter10.txt', 'w') as f:
    for pair in pairs:
        f.write(pair[2] + '\n')