def generate_nonterminals(start):
    terminals = []
    queue = [start]
    while queue:
        curr = queue.pop(0)
        poss_rules = [rule for rule in sent_rules if rule.lhs == curr]
        if poss_rules:
            rule = np.random.choice(poss_rules, p=[rule.prob for rule in poss_rules])
            queue = rule.rhs + queue
        else:
            terminals.append(curr)
    return terminals

def generate_words(terminals):
    words = []
    saw_prep = False
    for pos in terminals:
        if pos == 'P':
            saw_prep = True
        poss_words = [word for word in lexicon if word.pos == pos]
        if saw_prep and pos == 'N':
            poss_words = [Word('N', w) for w in location_nouns]
            saw_prep = False
        if poss_words:
            word = random.choice(poss_words)
            words.append(word.word)

    return words


def generate_sequence(start):
    terminals = generate_nonterminals(start)
    seq = generate_words(terminals)
    return ' '.join(seq)


def remove_pp(sent, rem_freq=0.2):
    output_seq = []
    saw_prep = False
    for word in sent.split(' '):
        if word in prepositions and np.random.uniform() <= rem_freq:
            saw_prep = True
        if not saw_prep:
            output_seq.append(word)
        if saw_prep and word in agent_nouns or word in location_nouns:
            saw_prep = False
    return ' '.join(output_seq)



def generate_act_pass_pair():
    np1 = generate_sequence('NP')
    np2 = generate_sequence('NP')
    while np1 == np2:
        np2 = generate_sequence('NP')

    verb_pair = random.choice(verb_pairs)
    
    active_sent = ' '.join([np1, verb_pair.active, np2]) + ' .'
    passive_sent = ' '.join([np2, verb_pair.passive, np1]) + ' .'

    passive_sent = remove_pp(passive_sent)
    passive_sent = substitute_synonyms(passive_sent)

    return (active_sent, passive_sent)

def generate_move_pp_pair():
    pp = generate_sequence('PP')
    sent = generate_sequence('S')

    s1 = ' '.join([pp, ',', sent]) + ' .'
    s2 = ' '.join([sent, pp]) + ' .'
    if np.random.uniform() > 0.7:
        s2 = sent + ' .'

    s2 = remove_pp(s2)
    s2 = substitute_synonyms(s2)

    return (s1, s2)



# def generate_modal_pair():
#     np1 = generate_sequence('NP')
#     np2 = generate_sequence('NP')
#     while np1 == np2:
#         np2 = generate_sequence('NP')

#     modal_pair = random.choice(modal_pairs)
#     infinitive = random.choice(infinitives)

#     seq1 = [np1, modal_pair.active, infinitive, np2]
#     seq2 = [np1, modal_pair.passive, infinitive, np2]

#     if np.random.uniform() > 0.5:
#         pp = generate_sequence('PP')
#         seq1 = [pp, ','] + seq1
#         if np.random.uniform() > 0.5:
#             seq2 = seq2 + [pp]


#     s1 = ' '.join(seq1) + ' .'
#     s2 = ' '.join(seq2) + ' .'

#     s2 = remove_pp(s2)
#     s2 = substitute_synonyms(s2, sub_freq=0.75)

#     return (s1, s2)
