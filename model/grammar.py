import random
import numpy as np 
import pickle
import nltk
from nltk.corpus import wordnet as wn
# from nltk.corpus import brown

class VerbPair:

    def __init__(self, active, passive):
        self.active = active
        self.passive = passive

class Word:

    def __init__(self, pos, word):
        self.pos = pos
        self.word = word

class Rule:
    def __init__(self, lhs, rhs, prob=1):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

class Constituent:
    def __init__(self, name, parent, children=[], is_leaf=False):
        self.name = name
        self.parent = parent
        self.children = children
        self.is_leaf = is_leaf

    def copy(self):
        """ Returns a copy of the tree from here down - sets parent to None """
        ccopies = []
        for child in self.children:
            ccopies.append(child.copy())
        return Constituent(self.name, None, ccopies, self.is_leaf)
        
    def bracketed_tree(self):
        ret = " ( " + self.name 
    
        for child in self.children:
            if child.is_leaf:
                ret += " {}".format(child.name)
            else:
                ret += child.bracketed_tree()
        ret += " )"
        return ret
        
    def sentence(self):
        ret = ""
        for child in self.children:
            if child.is_leaf:
                ret += "{} ".format(child.name)
            else:
                ret += child.sentence()
        if self.name == "S" and not self.parent:
            ret += "."
        return ret 



# word_freq = nltk.ConditionalFreqDist((tag, wrd.lower()) for wrd, tag in 
#         brown.tagged_words(tagset="universal"))

# with open('word_freq.pkl', 'wb') as f:
#     pickle.dump(word_freq, f)

with open('word_freq.pkl', 'rb') as f:
    word_freq = pickle.load(f)

sent_rules = [
    Rule('S', ['NP', 'VP'], 0.8),
    Rule('S', ['NP_prep', 'VP'], 0.2),
    Rule('NP', ['D', 'N'], 0.3),
    Rule('NP', ['D', 'Adj', 'N'], 0.7),
    Rule('NP_prep', ['NP', 'PP']),
    Rule('PP', ['P', 'NP_loc']),
    Rule('NP_loc', ['D', 'N_loc']),
    Rule('VP', ['V', 'NP'], 0.8),
    Rule('VP', ['V', 'NP_prep'], 0.2)
]

agent_nouns = ['cat', 'fish', 'sheep', 'moose', 'pig', 'rabbit', 'chicken', 'cow', 'duck', 'bird', 'crocodile', 'woman', 'man', \
    'boy', 'girl', 'student', 'professor', 'scientist', 'doctor', 'bear', 'penguin', 'dog', 'photographer', 'model', \
        'researcher', 'teacher']

location_nouns = ['house', 'barn', 'yard', 'kitchen', 'school', 'building', 'field', 'park', 'playground', 'room', 'market', 'supermarket', \
    'store', 'mall', 'stadium']

prepositions = ['in', 'by', 'near', 'at']

determiners = ['the', 'a']

adjectives = ['happy', 'silly', 'goofy', 'sleepy', 'big', 'small', 'tiny', 'curious', 'shy']

verb_pairs = [
    VerbPair('ate', 'was eaten by'),
    VerbPair('liked', 'was liked by'),
    VerbPair('loved', 'was loved by'),
    VerbPair('kicked', 'was kicked by'),
    VerbPair('bit', 'was bitten by'),
    VerbPair('tapped', 'was tapped by'),
    VerbPair('hated', 'was hated by'),
    VerbPair('disliked', 'was disliked by'),
    VerbPair('hunted', 'was hunted by'),
    VerbPair('stalked', 'was stalked by'),
    VerbPair('complimented', 'was complimented by')
]

modal_pairs = [
    VerbPair('must', 'needs to'),
    VerbPair('must', 'has got to'),
    VerbPair('needs to', 'ought to'),
    VerbPair('must', 'ought to'),
    VerbPair('needs to', 'should')
]

infinitives = [ 'eat', 'like', 'love', 'kick', 'bite', 'tap', 'hate', 'dislike', 'hunt', 'stalk', 'compliment']

active_verbs = [verb.active for verb in verb_pairs]
passive_verbs = [verb.passive for verb in verb_pairs]

verbs = active_verbs + passive_verbs

lexicon = [Word('N', w) for w in agent_nouns] + \
            [Word('N_loc', w) for w in location_nouns] + \
            [Word('V', w) for w in verbs] + \
            [Word('D', w) for w in determiners] + \
            [Word('Adj', w) for w in adjectives] + \
            [Word('P', w) for w in prepositions] + [Word(',', ',')]



def generate_tree(start, parent, lexicon):
    tree = Constituent(start, parent)
    poss_rules = [rule for rule in sent_rules if rule.lhs == start]
    if poss_rules:
        rule = np.random.choice(poss_rules, p=[rule.prob for rule in poss_rules])
        tree.children = [generate_tree(child, tree, lexicon) for child in rule.rhs] 
    else:
        poss_words = [w.word for w in lexicon if w.pos == start]
        word = np.random.choice(poss_words)
        child = Constituent(word, tree, children=[], is_leaf=True)
        tree.children = [child]
    return tree
    
def add_pp_front(sent):
    s = Constituent('S', None)
    sent.parent = s
    pp = generate_tree('PP', s, lexicon)
    s.children = [pp, Constituent(',', s, is_leaf=True), sent]
    return s

def add_pp_end(sent):
    s = Constituent('S', None)
    sent.parent = s
    pp = generate_tree('PP', s, lexicon)
    s.children = [sent, pp]
    return s

def remove_pp(tree, drop_p=0.5):
    if tree.name == 'PP' and np.random.uniform() <= drop_p:
        return
    new_children = []
    for child in tree.children:
        removed = remove_pp(child)
        if removed:
            new_children.append(removed)

    tree.children = new_children
    return tree

def most_common_syn(word, pos):
    synsets = wn.synsets(word, pos)
    max_freq = 0
    most_common = None
    if synsets:    
        if pos == wn.NOUN:
            tag = 'NOUN'
        else:
            tag = 'ADJ'
        for synset in synsets:
            synonym = synset.name().split('.')[0]
            syn_freq = word_freq[tag][synonym]
            if syn_freq > max_freq:
                max_freq = syn_freq
                most_common = synonym
    return most_common

def substitute_synonyms(sent, sub_freq=0.5):
    output_seq = []
    for word in sent.split(' '):
        output_word = word
        synonym = None
        if word in agent_nouns or word in location_nouns:
            synonym = most_common_syn(word, wn.NOUN)
        elif word in adjectives:
            synonym = most_common_syn(word, wn.ADJ)
        if synonym:
            sub = np.random.uniform()
            if sub <= sub_freq:
                output_word = synonym.replace('_', ' ')
        output_seq.append(output_word)
    return ' '.join(output_seq)

def generate_act_pass_pair():
    # generate tree using active verbs only
    act_tree = generate_tree('S', None, [w for w in lexicon if w.word not in passive_verbs])
    np1 = act_tree.children[0] # left child of S
    np2 = act_tree.children[1].children[1] # S -> NP VP, VP -> V NP

    act_verb = act_tree.children[1].children[0].children[0].name # S -> NP VP, VP -> V NP, V -> act_verb
    pass_verb = [pair.passive for pair in verb_pairs if pair.active == act_verb][0] # find matching passive

    # build the passive tree
    pass_tree = Constituent('S', None)
    pass_verb_tree = Constituent('VP', pass_tree)
    pass_v_tree = Constituent('V', pass_verb_tree)
    pass_v_tree.children = [Constituent(pass_verb, pass_v_tree, children=[], is_leaf=True)]
    pass_verb_tree.children = [pass_v_tree, np1]
    pass_tree.children = [np2, pass_verb_tree]

    return (act_tree, pass_tree)

def generate_move_pp_pair():
    sent1 = generate_tree('S', None, lexicon)
    sent2 = sent1.copy()

    s1 = Constituent('S', None)
    sent1.parent = s1
    pp = generate_tree('PP', s1, lexicon)
    s1.children = [pp, Constituent(',', s1, is_leaf=True), sent1]

    s2 = Constituent('S', None)
    sent2.parent = s2
    pp.parent = s2
    s2.children = [sent2, pp]
    
    return (s1, s2)

def generate_modal_pair():
    sent1 = generate_tree('S', None, lexicon)
    sent2 = sent1.copy()
    v1 = sent1.children[1].children[0]
    v2 = sent2.children[1].children[0]

    modal_pair = np.random.choice(modal_pairs)
    infinitive = np.random.choice(infinitives)

    v1.children = [Constituent(modal_pair.active + " " + infinitive, v1, is_leaf=True)]
    v2.children = [Constituent(modal_pair.passive + " " + infinitive, v2, is_leaf=True)]

    r = np.random.uniform()
    if r < 0.2:
        sent1 = add_pp_front(sent1)
    elif r < 0.4:
        sent1 = add_pp_end(sent1)
    
    return (sent1, sent2)


def generate_pairs(n, filename, pair_fns):
    with open(filename + '_pairs.txt', 'w') as f:
        with open(filename + '_ref_words.txt', 'w') as r:
            with open(filename + '_para_words.txt', 'w') as p:
                for _ in range(n):
                    pair_fn = random.choice(pair_fns)
                    s1, s2 = pair_fn()
                    if pair_fn == generate_move_pp_pair:
                        s2 = remove_pp(s2)
                        s2 = substitute_synonyms(s2.sentence())
                        f.write(s1.sentence() + '\t' + s2 + '\n')
                        r.write(s1.sentence() + '\n')
                        p.write(s2 + '\n')
                    else:
                        p1 = s1.sentence()
                        r.write(p1 + '\n')
                        p2 = s2.sentence()
                        r.write(p2 + '\n')

                        s2 = remove_pp(s2)
                        s2 = substitute_synonyms(s2.sentence())
                        p1 += '\t' + s2 + '\n'
                        p.write(s2 + '\n')

                        s1 = remove_pp(s1)
                        s1 = substitute_synonyms(s1.sentence())
                        p2 += '\t' + s1 + '\n'
                        p.write(s1 + '\n')
                        
                        f.write(p1)
                        f.write(p2)



# print(generate_act_pass_pair())
# generate_act_pass_pairs(10, 'pairs.txt')

pair_fns = [generate_act_pass_pair, generate_move_pp_pair, generate_modal_pair]
generate_pairs(160000, '../data/artificial_data/train', pair_fns)

# print(generate_move_pp_pair())
