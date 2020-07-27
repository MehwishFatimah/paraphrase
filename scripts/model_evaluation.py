''' Statistical model evaluation functions'''
import nltk
import pickle
import editdistance
from tqdm import tqdm
from nltk.corpus import stopwords


from nltk.corpus import wordnet as wn

with open('word_freq.pkl', 'rb') as f:
    word_freq = pickle.load(f)

stop_words = set(stopwords.words('english')) 

def read_seqs(filename):
    seq = []
    with open(filename) as f:
        for line in f:
            line = line.strip().split(' ')
            seq.append(line)
    return seq

def word_match(gold, output, allow_synonyms=True):
    ''' Checks if two words are the same or synonyms '''
    if gold != output:
        if allow_synonyms:
            pos = nltk.pos_tag([gold])[0][1]
            synsets = []
            if pos == 'N' or pos == 'NN':
                synsets = wn.synsets(gold, wn.NOUN)
            elif pos == 'JJ':
                synsets = wn.synsets(gold, wn.ADJ)
            synonyms = [synset.name().split('.')[0] for synset in synsets]
            synonyms = [word.replace('_', ' ') for word in synonyms]
            return output in synonyms
        else:
            return False
    return True

def item_for_item_accuracy(gold, output, allow_synonyms):
    ''' Returns the percentage of items in output that are in gold, allowing for synonyms if allow_synonyms is true '''
    correct_items = 0
    for gold_item, output_item in zip(gold, output):
        if allow_synonyms:
            if word_match(gold_item, output_item, allow_synonyms=True):
                correct_items += 1
        elif gold_item == output_item:
            correct_items += 1

    acc = correct_items * 1.0 / len(gold)
    return acc

def avg_item_for_item_accuracy(gold_seqs, output_seqs, allow_synonyms=False):
    ''' Returns the average item for item accuracy between two corpora.'''
    total_acc = 0
    for gold, output in zip(gold_seqs, output_seqs):
        total_acc += item_for_item_accuracy(gold, output, allow_synonyms)
    
    avg_acc = total_acc / len(gold_seqs)
    return avg_acc

def word_for_supertag_accuracy(gold_words, output_words, gold_stags, output_stags, allow_synonyms=False):
    ''' Returns the word-for-word accuracy for words that were assigned the correct supertag '''
    correct_words = 0
    correct_supertags = 0
    for gold_word, output_word, gold_stag, output_stag in zip(gold_words, output_words, gold_stags, output_stags):
        if gold_stag == output_stag:
            correct_supertags += 1
            if allow_synonyms:
                if word_match(gold_word, output_word, allow_synonyms=True):
                    correct_words += 1
            elif gold_word == output_word:
                correct_words += 1
    if correct_supertags == 0:
        return None
    acc = correct_words * 1.0 / correct_supertags
    return acc

def avg_word_for_supertag_accuracy(gold_sents, output_sents, gold_supertags, output_supertags, allow_synonyms=False):
    ''' Returns the average word-for-supertag accuracy between two corpora'''
    total_acc = 0
    total_sents = 0
    for gold_sent, output_sent, gold_stags, output_stags in zip(gold_sents, output_sents, gold_supertags, output_supertags):
        acc = word_for_supertag_accuracy(gold_sent, output_sent, gold_stags, output_stags, allow_synonyms)
        if acc is not None:
            total_acc += acc
            total_sents += 1

    avg_acc = total_acc / total_sents
    return avg_acc

# DATA_DIR = 'linear-hierarchical-experiment/test/'
# OUTPUT_DIR = 'linear-hierarchical-experiment/model-outputs/unidirectional-256-2/'
DATA_DIR = 'data/' # gold data
OUTPUT_DIR = 'data/' # model output
PREFIX = 'test'

# ref_sents = read_seqs(DATA_DIR + PREFIX + '-ref-words.txt')
para_sents = read_seqs(DATA_DIR + PREFIX + '-para-words.txt')
output_sents = read_seqs(OUTPUT_DIR + PREFIX + '-bidir-100.txt')

para_supertags = read_seqs(DATA_DIR + PREFIX + '-para-supertags.txt')
output_supertags = read_seqs(OUTPUT_DIR + PREFIX + '-bidir-100-supertags.txt')


print('Average word-for-word accuracy: ', avg_item_for_item_accuracy(para_sents, output_sents))
print('Average word-for-word accuracy allowing synonyms:', avg_item_for_item_accuracy(para_sents, output_sents, allow_synonyms=True))
print('Average supertag-for-supertag accuracy: ', avg_item_for_item_accuracy(para_supertags, output_supertags))
print('Of correct supertags, average word accuracy: ', avg_word_for_supertag_accuracy(para_sents, output_sents, para_supertags, output_supertags))
print('Of correct supertags, average word accuracy allowing synonyms:', avg_word_for_supertag_accuracy(para_sents, output_sents, para_supertags, output_supertags, allow_synonyms=True))
