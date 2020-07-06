""" Implements a probabilistic lexicalized CFG for paraphrase generation.  
    Paraphrase operations include: 
        - passivization 
        - PP movement
        - subject and object clefts
        - ditransitive constructions
        - dative shift
"""

import numpy as np
import pickle
import nltk
from nltk.corpus import wordnet as wn

##### CLASSES #####

class Tree:
    
    def __init__(self, name, parent=None, children=[], is_leaf=False):
        ''' Represents a syntactic tree for a sentence '''
        self.name = name
        self.parent = parent
        self.children = children
        self.is_leaf = is_leaf

    def copy(self, parent=None):
        """ Returns a copy of the tree from here down - sets parent to None """
        copies = []
        for child in self.children:
            copies.append(child.copy())
        return Tree(self.name, parent, copies, self.is_leaf)
        
    def bracketed_tree(self):
        """ Returns the tree in bracketed notation """ 
        ret = " ( " + self.name 
    
        for child in self.children:
            if child.is_leaf:
                ret += " {}".format(child.name)
            else:
                ret += child.bracketed_tree()
        ret += " )"
        return ret
        
    def sentence(self):
        """ Returns the sentence represented by this tree as a string separated by spaces """
        ret = ""
        for child in self.children:
            if child.is_leaf:
                ret += "{} ".format(child.name)
            else:
                ret += child.sentence()
        if self.name == "S" and not self.parent:
            ret += "."
        return ret 


class Rule:
    """ A CFG production rule, with left hand side, right hand side, and probability """
    def __init__(self, lhs, rhs, prob=1):
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

class Word:
    """ A word and its associated POS """
    def __init__(self, pos, word):
        self.pos = pos
        self.word = word


##### GLOBAL VARIABLES #####
CFG_RULES = [
    Rule('NP', ['D', 'N_agent'], 0.6),
    Rule('NP', ['D', 'Adj', 'N_agent'], 0.2),
    Rule('NP', ['D', 'N_agent', 'PP'], 0.1),
    Rule('NP', ['D', 'Adj', 'N_agent', 'PP'], 0.1),

    Rule('PP', ['P', 'NP_loc']),
    
    Rule('PP_dat', ['P_to', 'NP']),

    Rule('NP_loc', ['D', 'N_loc']),
    
    Rule('NP_obj', ['D', 'N_obj'], 0.5),
    Rule('NP_obj', ['D', 'Adj_obj', 'N_obj'], 0.5),

    Rule('VP_trans', ['V_trans', 'NP'], 0.8),
    Rule('VP_trans', ['V_trans', 'NP', 'PP'], 0.2),

    Rule('VP_ditrans', ['V_ditrans', 'NP_obj', 'PP_dat'], 0.5),
    Rule('VP_ditrans', ['V_ditrans', 'NP', 'NP_obj'], 0.5)
]

agent_nouns = ['cat', 'fish', 'sheep', 'moose', 'pig', 'rabbit', 'chicken', 'cow', 'duck', 'bird', 'crocodile', 'woman', 'man', \
    'boy', 'girl', 'student', 'professor', 'scientist', 'doctor', 'bear', 'penguin', 'dog', 'photographer', 'model', \
        'researcher', 'teacher']

object_nouns = ['letter', 'gift', 'package', 'box', 'book', 'candle', 'note', 'card', 'present']

location_nouns = ['house', 'barn', 'yard', 'kitchen', 'school', 'building', 'field', 'park', 'playground', 'room', 'market', 'supermarket', \
    'shop', 'mall', 'stadium']

prepositions = ['in', 'near', 'at', 'next to', 'close to']

determiners = ['the', 'a']

adjectives = ['happy', 'silly', 'goofy', 'sleepy', 'big', 'small', 'tiny', 'curious', 'shy']
object_adjectives = ['blue', 'beautiful', 'lovely', 'white', 'kind', 'generous']


transitive_verbs = {
    'eats': 'eaten',
    'likes': 'liked',
    'loves': 'loved',
    'kicks': 'kicked',
    'taps': 'tapped',
    'hates': 'hated', 
    'dislikes': 'disliked',
    'pushes': 'pushed',
    'hunts': 'hunted',
    'stalks': 'stalked',
    'compliments': 'complimented'
}

ditransitive_verbs = {
    'gives': 'given',
    'sends': 'sent',
    'mails': 'mailed',
    'lends': 'lent',
    'passes': 'passed',
    'promises': 'promised',
    'sells': 'sold'
}

passive_safe_ditransitive_verbs = ['sends', 'lends', 'passes', 'sells']

LEXICON = [Word('N_agent', w) for w in agent_nouns] + \
            [Word('N_loc', w) for w in location_nouns] + \
            [Word('N_obj', w) for w in object_nouns] + \
            [Word('V_trans', w) for w in transitive_verbs] + \
            [Word('V_ditrans', w) for w in ditransitive_verbs] + \
            [Word('D', w) for w in determiners] + \
            [Word('Adj', w) for w in adjectives] + \
            [Word('Adj_obj', w) for w in object_adjectives] + \
            [Word('P', w) for w in prepositions] + \
            [Word('P_to', 'to')] + [Word(',', ',')]

with open('word_freq.pkl', 'rb') as f:
    word_freq = pickle.load(f)

##### FUNCTIONS #####
def generate_tree(start, parent, sent_rules=CFG_RULES, lexicon=LEXICON):
    """ Generates a Tree object using the given start symbol, parent, CFG rules, and lexicon """
    tree = Tree(start, parent)
    poss_rules = [rule for rule in sent_rules if rule.lhs == start]
    if poss_rules:
        rule = np.random.choice(poss_rules, p=[rule.prob for rule in poss_rules])
        tree.children = [generate_tree(child, tree, sent_rules, lexicon) for child in rule.rhs] 
    else:
        poss_words = [w.word for w in lexicon if w.pos == start]
        word = np.random.choice(poss_words)
        child = Tree(word, tree, children=[], is_leaf=True)
        tree.children = [child]
    return tree

def generate_trans_act_pass_pair():
    """ the boy compliments the moose -> the moose is complimented by the boy """
    # Basic active transitive form is 'NP1 V_trans NP2'
    act_tree = Tree('S')
    np1 = generate_tree('NP', act_tree)
    act_vp = Tree('VP', act_tree)
    act_verb = np.random.choice([v for v in transitive_verbs])
    act_verb_tree = Tree(act_verb, act_vp, is_leaf=True)
    np2 = generate_tree('NP', act_vp)
    while np2.sentence() == np1.sentence():
        np2 = generate_tree('NP', act_vp)
    act_vp.children = [act_verb_tree, np2] # VP -> V_trans NP
    act_tree.children = [np1, act_vp] # S -> NP VP

    # Basic passive transitive form is 'NP2 is V_trans_pass by NP1'
    pass_tree = Tree('S')
    pass_vp = Tree('VP', pass_tree)
    pass_verb = transitive_verbs[act_verb]
    pass_verb_tree = Tree('is ' + pass_verb + ' by', pass_vp, is_leaf=True)
    pass_np1 = np2.copy(parent=pass_tree)
    pass_np2 = np1.copy(parent=pass_vp)
    pass_vp.children = [pass_verb_tree, pass_np2]
    pass_tree.children = [pass_np1, pass_vp]

    return (act_tree, pass_tree)

def generate_ditrans_act_pass_pair():
    """ the boy gives the moose a gift -> a gift is given to the moose by the boy OR
        the boy gives a gift to the moose -> a gift is given to the moose by the boy """
    act_tree = Tree('S')
    np_agent = generate_tree('NP', act_tree)
    act_vp = Tree('VP', act_tree)
    act_verb = np.random.choice([v for v in passive_safe_ditransitive_verbs])

    act_verb_tree = Tree(act_verb, act_vp, is_leaf=True)
    np_object = generate_tree('NP_obj', act_vp)
    pp_target = generate_tree('PP_dat', act_vp)
    
    
    act_vp.children = [act_verb_tree, np_object, pp_target]
    
    act_tree.children = [np_agent, act_vp]

    pass_tree = Tree('S')
    pass_object = np_object.copy(parent=pass_tree)
    pass_vp = Tree('VP', pass_tree)
    pass_verb = ditransitive_verbs[act_verb]
    pass_verb_tree = Tree('is ' + pass_verb, pass_vp, is_leaf=True)
    pass_target = pp_target.copy(parent=pass_vp)
    
    pass_agent_tree = Tree('PP', pass_vp)
    pass_agent = np_agent.copy(parent=pass_agent_tree)
    by_pp = Tree('by', pass_agent_tree, is_leaf=True)
    pass_agent_tree.children = [by_pp, pass_agent]

    pass_vp.children = [pass_verb_tree, pass_target, pass_agent_tree]
    pass_tree.children = [pass_object, pass_vp]

    return (act_tree, pass_tree)

def generate_ditrans_dative_pair():
    ''' The man gives the girl a book -> the man gives a book to the girl '''
    dat_tree = Tree('S')
    dat_agent = generate_tree('NP', dat_tree)

    ind_tree = Tree('S')
    ind_agent = dat_agent.copy(parent=ind_tree)

    dat_vp = Tree('VP', dat_tree)
    ind_vp = Tree('VP', ind_tree)
    
    # verb = np.random.choice([v for v in ditransitive_verbs])
    verb = 'gives'
    dat_verb_tree = Tree(verb, dat_vp, is_leaf=True)
    ind_verb_tree = Tree(verb, ind_vp, is_leaf=True)

    dat_object = generate_tree('NP_obj', dat_vp)
    ind_object = dat_object.copy(parent=ind_vp)

    ind_target = generate_tree('PP_dat', ind_vp)
    np_target = ind_target.children[1]
    dat_target = np_target.copy(parent=dat_vp)
    
    dat_vp.children = [dat_verb_tree, dat_target, dat_object]
    ind_vp.children = [ind_verb_tree, ind_object, ind_target]
    
    dat_tree.children = [dat_agent, dat_vp]
    ind_tree.children = [ind_agent, ind_vp]

    return (dat_tree, ind_tree)

def generate_trans_subj_it_cleft_pair(active=True):
    ''' the sheep complimented the cow -> it was the sheep who complimented the cow OR 
        the cow was complimented by the sheep -> it was the cow who was complimented by the sheep'''
    reg_tree = Tree('S')
    it_tree = Tree('S')

    reg_agent = generate_tree('NP', reg_tree)
    
    reg_vp = Tree('VP', reg_tree)
    it_vp = Tree('VP', it_tree)

    verb = np.random.choice([v for v in transitive_verbs])
    if active: 
        reg_verb = Tree(verb, reg_vp, is_leaf=True)
        it_verb = Tree(verb, it_vp, is_leaf=True)
    else:
        verb = transitive_verbs[verb]
        reg_verb = Tree('is ' + verb + ' by', reg_vp, is_leaf=True)
        it_verb = Tree('is ' + verb + ' by', it_vp, is_leaf=True)

    reg_target = generate_tree('NP', reg_vp)
    while reg_target.sentence() == reg_agent.sentence():
        reg_target = generate_tree('NP', reg_vp)
    it_target = reg_target.copy(parent=it_vp)

    reg_vp.children = [reg_verb, reg_target]
    it_vp.children = [it_verb, it_target]

    cleft_tree = Tree('IT', it_tree)
    it_agent = reg_agent.copy(parent=cleft_tree)
    it_cleft = Tree('it is', cleft_tree, is_leaf=True)
    wh = Tree('that', cleft_tree, is_leaf=True)
    cleft_tree.children = [it_cleft, it_agent, wh]

    reg_tree.children = [reg_agent, reg_vp]
    it_tree.children = [cleft_tree, it_vp]

    return (reg_tree, it_tree)

def generate_trans_obj_it_cleft_pair():
    ''' the sheep complimented the cow -> it was the cow that the sheep complimented '''
    reg_tree = Tree('S')
    it_tree = Tree('S')

    reg_agent = generate_tree('NP', reg_tree)
    it_agent = reg_agent.copy(parent=it_tree)

    reg_vp = Tree('VP', reg_tree)
    it_vp = Tree('VP', it_tree)

    verb = np.random.choice([v for v in transitive_verbs])
    reg_verb = Tree(verb, reg_vp, is_leaf=True)
    it_verb = Tree(verb, it_vp, is_leaf=True)

    reg_target = generate_tree('NP', reg_vp)
    while reg_target.sentence() == reg_agent.sentence():
        reg_target = generate_tree('NP', reg_vp)

    reg_vp.children = [reg_verb, reg_target]
    it_vp.children = [it_verb]

    cleft_tree = Tree('IT', it_tree)
    it_target = reg_target.copy(parent=cleft_tree)
    it_cleft = Tree('it is', cleft_tree, is_leaf=True)
    wh = Tree('that', cleft_tree, is_leaf=True)
    cleft_tree.children = [it_cleft, it_target, wh]

    reg_tree.children = [reg_agent, reg_vp]
    it_tree.children = [cleft_tree, it_agent, it_vp]

    return (reg_tree, it_tree)

def generate_trans_obj_wh_cleft_pair(active=True):
    ''' Generates a transitive pair with object wh-cleft'''
    reg_tree = Tree('S')
    wh_tree = Tree('S')

    reg_agent = generate_tree('NP', reg_tree)
    wh_agent = reg_agent.copy(parent=wh_tree)

    reg_vp = Tree('VP', reg_tree)
    wh_vp = Tree('VP', wh_tree)

    verb = np.random.choice([v for v in transitive_verbs])
    if active: 
        reg_verb = Tree(verb, reg_vp, is_leaf=True)
        wh_verb = Tree(verb, wh_vp, is_leaf=True)
    else:
        verb = transitive_verbs[verb]
        reg_verb = Tree('is ' + verb + ' by', reg_vp, is_leaf=True)
        wh_verb = Tree('is ' + verb + ' by', wh_vp, is_leaf=True)

    reg_target = generate_tree('NP', reg_vp)
    while reg_target.sentence() == reg_agent.sentence():
        reg_target = generate_tree('NP', reg_vp)
    wh_target = reg_target.copy(parent=wh_vp)

    reg_vp.children = [reg_verb, reg_target]
    reg_tree.children = [reg_agent, reg_vp]

    wh_vp.children = [wh_verb, wh_target, Tree('is', wh_vp, is_leaf=True), wh_agent]
    wh_subj = Tree('what', wh_tree, is_leaf=True)
    wh_tree.children = [wh_subj, wh_vp]

    return (reg_tree, wh_tree)

def generate_trans_move_pp_pair(active=True):
    ''' Generates a transitive pair with PP movement'''
    tree = Tree('S')
    np1 = generate_tree('NP', tree)
    vp = Tree('VP', tree)
    verb = np.random.choice([v for v in transitive_verbs])
    verb_tree = Tree(verb, vp, is_leaf=True)
    if not active:
        verb = transitive_verbs[verb]
        verb_tree = Tree('is ' + verb + ' by', vp, is_leaf=True)
    
    np2 = generate_tree('NP', vp)
    while np2.sentence() == np1.sentence():
        np2 = generate_tree('NP', vp)
    pp = generate_tree('PP', vp)
    vp.children = [verb_tree, np2, pp] # VP -> V_trans NP
    tree.children = [np1, vp] # S -> NP VP
    
    move_tree = Tree('S')
    move_vp = Tree('VP', move_tree)
    move_np1 = np1.copy(move_tree)
    move_verb_tree = verb_tree.copy(move_vp)
    move_np2 = np2.copy(move_vp)
    move_vp.children = [move_verb_tree, move_np2]
    move_pp_tree = Tree('PP', move_tree)
    move_pp = pp.copy(move_pp_tree)
    move_comma = Tree(',', move_pp_tree, is_leaf=True)
    move_pp_tree.children = [move_pp, move_comma]
    move_tree.children = [move_pp_tree, move_np1, move_vp]

    return tree, move_tree

def generate_trans_voice_move_pp_pair(start_active=True):
    ''' Generates a pair that changes voice and moves a PP '''
    tree = Tree('S')
    np1 = generate_tree('NP', tree)
    vp = Tree('VP', tree)
    act_verb = np.random.choice([v for v in transitive_verbs])
    verb_tree = Tree(act_verb, vp, is_leaf=True)
    if not start_active:
        pass_verb = transitive_verbs[act_verb]
        verb_tree = Tree('is ' + pass_verb + ' by', vp, is_leaf=True)
    
    np2 = generate_tree('NP', vp)
    while np2.sentence() == np1.sentence():
        np2 = generate_tree('NP', vp)
    pp = generate_tree('PP', vp)
    vp.children = [verb_tree, np2, pp] # VP -> V_trans NP
    tree.children = [np1, vp] # S -> NP VP
    
    move_tree = Tree('S')
    move_vp = Tree('VP', move_tree)
    move_np1 = np2.copy(move_tree)
    if start_active:
        move_verb = transitive_verbs[act_verb]
        move_verb_tree = Tree('is ' + move_verb + ' by', move_vp, is_leaf=True)
    else:
        move_verb_tree = Tree(act_verb, move_vp, is_leaf=True)

    move_np2 = np1.copy(move_vp)
    move_vp.children = [move_verb_tree, move_np2]
    move_pp_tree = Tree('PP', move_tree)
    move_pp = pp.copy(move_pp_tree)
    move_comma = Tree(',', move_pp_tree, is_leaf=True)
    move_pp_tree.children = [move_pp, move_comma]
    move_tree.children = [move_pp_tree, move_np1, move_vp]

    return tree, move_tree

def generate_ditrans_move_pp_pair(start_dative=True, shift=False):
    ''' Generates a ditransitive pair with PP movement, with the option to use dative shift as well '''
    reg_tree = Tree('S')
    reg_agent = generate_tree('NP', reg_tree)

    move_tree = Tree('S')
    move_agent = reg_agent.copy(parent=move_tree)

    reg_vp = Tree('VP', reg_tree)
    move_vp = Tree('VP', move_tree)
    
    verb = np.random.choice([v for v in ditransitive_verbs])
    if start_dative or (not start_dative and shift):
        verb = 'gives'
    reg_verb_tree = Tree(verb, reg_vp, is_leaf=True)
    move_verb_tree = Tree(verb, move_vp, is_leaf=True)

    reg_object = generate_tree('NP_obj', reg_vp)
    move_object = reg_object.copy(parent=move_vp)

    ind_target = generate_tree('PP_dat', reg_vp)
    np_target = ind_target.children[1]
    if start_dative:
        reg_target = np_target.copy(parent=reg_vp)
        if shift:
            move_target = ind_target.copy(move_vp)
    else:
        reg_target = ind_target
        if shift:
            move_target = np_target.copy(move_vp)
    if not shift:
        move_target = reg_target.copy(move_vp)

    
    reg_pp = generate_tree('PP', reg_vp)
    if start_dative:
        reg_vp.children = [reg_verb_tree, reg_target, reg_object, reg_pp]
        if shift:
            move_vp.children = [move_verb_tree, move_object, move_target]
        else:
            move_vp.children = [move_verb_tree, move_target, move_object]
    else:
        reg_vp.children = [reg_verb_tree, reg_object, reg_target, reg_pp]
        if shift:
            move_vp.children = [move_verb_tree, move_target, move_object]
        else:
            move_vp.children = [move_verb_tree, move_object, move_target]
    
    
    move_pp_tree = Tree('PP', move_tree)
    move_pp = reg_pp.copy(move_pp_tree)
    move_comma = Tree(',', move_pp_tree, is_leaf=True)
    move_pp_tree.children = [move_pp, move_comma]
    
    reg_tree.children = [reg_agent, reg_vp]
    move_tree.children = [move_pp_tree, move_agent, move_vp]

    return (reg_tree, move_tree)

def remove_pp(tree):
    ''' removes PPs from a tree '''
    if tree.name == 'PP':
        return
    new_children = []
    for child in tree.children:
        removed = remove_pp(child)
        if removed:
            new_children.append(removed)

    tree.children = new_children
    return tree

def remove_adj(tree):
    ''' removes adjectives from a tree '''
    if tree.name in adjectives or tree.name in object_adjectives:
        return
    new_children = []
    for child in tree.children:
        removed = remove_adj(child)
        if removed:
            new_children.append(removed)
    
    tree.children = new_children
    return tree

def most_common_syn(word, pos):
    ''' finds the most common synonym of a word with the given POS'''
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
    ''' substitutes noun or adj synonyms with sub_freq probability'''
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

def write_to_files(s1, s2, master, ref, para, one_dir=False):
    ''' writes paraphrase pairs to files, in reverse order as well if one_dir=False'''
    s1 = s1.sentence()
    try:
        s2 = s2.sentence()
    except:
        pass
    master.write(s1 + '\t' + s2 + '\n')
    ref.write(s1 + '\n')
    para.write(s2 + '\n')

    if not one_dir:
        master.write(s2 + '\t' + s1 + '\n')
        ref.write(s2 + '\n')
        para.write(s1 + '\n')

def sub_rem_write(sent, master, ref, para):
    ''' Attempt synonym substitution, adjective removal, and PP removal separately and write to files if successful'''
    syn = substitute_synonyms(sent.sentence(), sub_freq=1.0)
    if syn != sent.sentence():
        write_to_files(sent, syn, master, ref, para)
    rem_adj = remove_adj(sent)
    if rem_adj.sentence() != sent.sentence():
        write_to_files(sent, rem_adj, master, ref, para, one_dir=True)
    rem_pp = remove_pp(sent)
    if rem_pp.sentence() != sent.sentence():
        write_to_files(sent, rem_pp, master, ref, para, one_dir=True)

def generate_sentence_set(master_filename, ref_filename, para_filename, n=1):
    ''' Generates n base sentence pairs with several modifications of each (reverse order, substitute synonyms, remove phrases)'''
    with open(master_filename, 'w') as master:
        with open(ref_filename, 'w') as ref:
            with open(para_filename, 'w') as para:
                for _ in range(n):
                    a, b = generate_trans_act_pass_pair()
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_trans_act_pass_pair()
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_ditrans_act_pass_pair()
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_ditrans_act_pass_pair()
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_ditrans_dative_pair()
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_ditrans_dative_pair()
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_trans_move_pp_pair(active=True)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_trans_move_pp_pair(active=True)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_trans_move_pp_pair(active=False)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_trans_move_pp_pair(active=False)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_trans_voice_move_pp_pair(start_active=True)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_trans_voice_move_pp_pair(start_active=True)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_trans_voice_move_pp_pair(start_active=False)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_trans_voice_move_pp_pair(start_active=False)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_ditrans_move_pp_pair(start_dative=True, shift=False)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_ditrans_move_pp_pair(start_dative=True, shift=False)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_ditrans_move_pp_pair(start_dative=False, shift=False)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_ditrans_move_pp_pair(start_dative=False, shift=False)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)
                    
                    a, b = generate_ditrans_move_pp_pair(start_dative=True, shift=True)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_ditrans_move_pp_pair(start_dative=True, shift=True)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)

                    a, b = generate_ditrans_move_pp_pair(start_dative=False, shift=True)
                    while len(a.sentence().split(' ')) >= 15 or len(b.sentence().split(' ')) >= 15:
                        a, b = generate_ditrans_move_pp_pair(start_dative=False, shift=True)
                    write_to_files(a, b, master, ref, para)
                    sub_rem_write(a, master, ref, para)
                    sub_rem_write(b, master, ref, para)



if __name__ == "__main__":
    prefix = 'new-data/' + 'test/'
    generate_sentence_set(prefix + 'test-pairs.txt', prefix + 'test-ref.txt', prefix + 'test-para.txt', n=1000)

    # a, b = generate_trans_act_pass_pair()
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_ditrans_act_pass_pair(use_dative=True)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_ditrans_act_pass_pair(use_dative=False)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_ditrans_dative_pair()
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_move_pp_pair(active=True)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_move_pp_pair(active=False)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_voice_move_pp_pair(start_active=True)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_voice_move_pp_pair(start_active=False)
    # print(a.sentence())
    # print(b.sentence())

    # print(substitute_synonyms(a.sentence(), sub_freq=1.0))

    # a, b = generate_ditrans_move_pp_pair(start_dative=True, shift=False)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_ditrans_move_pp_pair(start_dative=False, shift=False)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_ditrans_move_pp_pair(start_dative=True, shift=True)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_ditrans_move_pp_pair(start_dative=False, shift=True)
    # print(a.sentence())
    # print(b.sentence())

    # print(remove_pp(b).sentence())
    # print(remove_adj(b).sentence())

    # a, b = generate_trans_subj_it_cleft_pair(active=True)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_subj_it_cleft_pair(active=False)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_obj_it_cleft_pair()
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_obj_wh_cleft_pair(active=True)
    # print(a.sentence())
    # print(b.sentence())

    # a, b = generate_trans_obj_wh_cleft_pair(active=False)
    # print(a.sentence())
    # print(b.sentence())
