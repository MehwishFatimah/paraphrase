''' Implements a variety of scoring metrics for model evaluation '''
from rouge import Rouge, FilesRouge
import numpy as np
import nltk
import pickle
import pprint
from nltk.corpus import wordnet as wn
from grammar import get_synonym_list, prepositions, modals, verbs

DIR = 'uni'
HIDDEN_SIZE = '256'
TYPE = 'hierarchical'
MICA_WEIGHT = 0.5
ROUGE_WEIGHT = 0.5
USE_ROLE_WEIGHTS = True
ROLES = ['Root', '0', '1']
ALLOWED_ADJ = ['in', 'by', 'near', 'at']
# ALLOWED_ADJ = []

DATA_DIR = '../data/artificial-data/set-2/test/'
REFERENCE_DIR = 'linear-hierarchical-experiment/test/'
OUTPUT_DIR = 'linear-hierarchical-experiment/model-outputs/' + DIR + 'directional-' + HIDDEN_SIZE +'/'
PREFIX = 'test-' + DIR + '-' + HIDDEN_SIZE + '-' 


rouge = Rouge()
synonyms = get_synonym_list()

def get_rouge_score(ref, hyp):
    ''' Returns f-score of rouge-l for a pair of sentences'''
    scores = rouge.get_scores(hyp, ref)[0]
    return scores['rouge-l']['p'], scores['rouge-l']['r'], scores['rouge-l']['f']

def get_mica_score(ref, hyp):
    ''' Returns proportion of dependency relations in common between ref, hyp represented as sets of 3-tuples: root, target, relation '''
    overlap = ref.intersection(hyp)
    precision = len(overlap) * 1.0 / len(hyp)
    recall = len(overlap) * 1.0 / len(ref)
    fscore = 2 * precision * recall / (precision + recall)
    return fscore

def get_weighted_mica_score(ref, hyp):
    ''' Returns a precision, recall, fscore for MICA using only the relations and weights given. '''
    ref_subset = filter_triples(ref)
    hyp_subset = filter_triples(hyp)
    # print(ref_subset)
    # print(hyp_subset)
    
    precision = 0.0
    recall = 0.0
    fscore = 0.0
    
    if ref_subset:
        overlap = intersect(ref_subset, hyp_subset)
        if hyp_subset:
            precision = len(overlap) * 1.0 / len(hyp_subset)
        recall = len(overlap) * 1.0 / len(ref_subset)
        if precision and recall:
            fscore = 2 * precision * recall / (precision + recall)

        return precision, recall, fscore
    else:
        return None, None, None

def get_mica_by_relation(ref_triples, hyp_triples):
    results = {}
    for role in ROLES:
        ref_subset = filter_triples(ref_triples, roles=[role])
        hyp_subset = filter_triples(hyp_triples, roles=[role])

        precision = 0.0
        recall = 0.0
        fscore = 0.0
        # print(ref_subset, hyp_subset)
        if ref_subset:
            overlap = intersect(ref_subset, hyp_subset)
            if hyp_subset:
                precision = len(overlap) * 1.0 / len(hyp_subset)
            recall = len(overlap) * 1.0 / len(ref_subset)
            if precision and recall:
                fscore = 2 * precision * recall / (precision + recall)
        
            results[role] = {
                'precision': precision,
                'recall': recall,
                'fscore': fscore
            }
    return results

def get_average_mica_by_relation(refs, hyps):
    measures = ['precision', 'recall', 'fscore']
    averages = {
        'Root': {
            'precision': [],
            'recall': [],
            'fscore': []
        },
        '0': {
            'precision': [],
            'recall': [],
            'fscore': []
        },
        '1': {
            'precision': [],
            'recall': [],
            'fscore': []
        },
        'Adj': {
            'precision': [],
            'recall': [],
            'fscore': []
        },
    }
    for ref_triples, hyp_triples in zip(refs, hyps):
        results = get_mica_by_relation(ref_triples, hyp_triples)
        for role in ROLES:
            if role in results:
                for measure in measures:
                    averages[role][measure].append(results[role][measure])
    for role in ROLES:
        for measure in measures:
            averages[role][measure] = np.mean(averages[role][measure])
    
    return averages


def get_paraphrase_score(ref_sent, hyp_sent, ref_triples, hyp_triples, weighted_mica=False):
    ''' Returns weighted sum of ROUGE and MICA scores for a pair of sentences '''
    # higher ROUGE -> more similar -> worse paraphrase
    rouge_score = get_rouge_score(ref_sent, hyp_sent)
    # higher MICA score -> more semantic similarity -> better paraphrase
    if weighted_mica:
        mica_score = get_weighted_mica_score(ref_triples, hyp_triples)
    else:
        mica_score = get_mica_score(ref_triples, hyp_triples)
    # weighted sum, flipping rouge score
    paraphrase_score = MICA_WEIGHT * mica_score + ROUGE_WEIGHT * (1 - rouge_score)
    return paraphrase_score, mica_score, 1 - rouge_score

def get_average_paraphrase_score(refs, hyps, ref_triples, hyp_triples, weighted_mica=False):
    ''' Returns average paraphrase, mica, and inverse-ROUGE score for set of reference and hypothesis sentences'''
    total_p = 0
    total_m = 0
    total_r = 0
    for ref, hyp, ref_trip, hyp_trip in zip(refs, hyps, ref_triples, hyp_triples):
        p, m, r = get_paraphrase_score(ref, hyp, ref_trip, hyp_trip, weighted_mica=weighted_mica)
        total_p += p
        total_m += m
        total_r += r
    return total_p / len(refs), total_m / len(refs), total_r / len(refs)

def get_average_rouge(refs, hyps):
    precisions = []
    recalls = []
    fscores = []
    for ref, hyp in zip(refs, hyps):
        p, r, f = get_rouge_score(ref, hyp)
        precisions.append(p)
        recalls.append(r)
        fscores.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(fscores)

def get_average_mica(ref_triples, hyp_triples):
    precisions = []
    recalls = []
    fscores = []
    for ref, hyp in zip(ref_triples, hyp_triples):
        p, r, f = get_weighted_mica_score(ref, hyp)
        if p is not None and r is not None and f is not None:
            precisions.append(p)
            recalls.append(r)
            fscores.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(fscores)

def get_mica_triples_from_lines(sent_lines):
    ''' Reads lines of MICA output into triples of root, target, relation and returns a set'''
    sent_triples = set()
    for line in sent_lines:
        l, attributes = line.strip().split('||')
        l = l.split(' ')
        attributes = [att for att in attributes.split(' ') if att]
        target = l[1]
        parent = l[4]
        relation = None
        
        for item in attributes:
            att, val = item.split(':')
            if att == 'DRole':
                relation = val
        if relation:
            triple = (parent, target, relation)
            sent_triples.add(triple)
    return sent_triples

def get_mica_triples(filename):
    ''' Gets the set of MICA triples for each sentence in a MICA output file and returns the list of sets'''
    with open(filename, 'r') as input_file:
        triples = []
        line = input_file.readline()
        sent_lines = []
        while line != '':
            if line[0] == '#':
                sent_triples = get_mica_triples_from_lines(sent_lines)
                triples.append(sent_triples)
                sent_lines = []
                line = input_file.readline()
                line = input_file.readline()
                line = input_file.readline()
                line = input_file.readline()
                line = input_file.readline()
            else:
                sent_lines.append(line)
                line = input_file.readline()
    return triples

def read_seqs(filename):
    ''' Returns list of sentences from file '''
    seq = []
    with open(filename) as f:
        for line in f:
            line = line.strip()
            seq.append(line)
    return seq

def filter_triples(triples, roles=ROLES):
    result = set()
    for t in triples:
        if t[2] in roles:
            if t[2] == 'Adj':
                if t[1] in ALLOWED_ADJ:
                    result.add(t)
            elif t[2] == '1':
                if t[0] not in prepositions and t[1] not in prepositions:
                    result.add(t)
            else:
                result.add(t)
    return result

def intersect(ref, hyp, allow_synonyms=True):
    intersection = set()
    for t1 in ref:
        for t2 in hyp:
            if not allow_synonyms:
                if t1 == t2:
                    intersection.add(t1)
            else:
                if t1[2] == t2[2]:
                    if is_synonym(t2[0], t1[0]) and is_synonym(t2[1], t1[1]):
                        intersection.add(t1)
    return intersection

def get_root_0_1(triples):
    sent_root = None
    sent_0 = None
    sent_1 = None
    sent_modal = None
    for triple in triples:
        if triple[2] == 'Root':
            if triple[0] in modals:
                sent_modal = triple[0]
                for t in triples:
                    if t[0] == triple[0] and t[1] in verbs:
                        sent_root = t[1]
            else:
                sent_root = triple[0]
    for triple in triples:
        if triple[2] == '0' and (triple[0] == sent_root or triple[0] == sent_modal):
            sent_0 = triple[1]
        elif triple[2] == '1' and triple[0] == sent_root:
            sent_1 = triple[1]

    return sent_root, sent_0, sent_1

def mica_overlap(ref, hyp, allow_synonyms=True):
    ref_root, ref_0, ref_1 = get_root_0_1(ref)
    hyp_root, hyp_0, hyp_1 = get_root_0_1(hyp)

    return is_synonym(hyp_root, ref_root), is_synonym(hyp_0, ref_0), is_synonym(hyp_1, ref_1)

def average_mica_overlap(refs, hyps):
    roots = []
    zeros = []
    ones = []
    for ref, hyp in zip(refs, hyps):
        root, zero, one = mica_overlap(ref, hyp)
        if root is not None:
            roots.append(int(root))
        if zero is not None:
            zeros.append(int(zero))
        if one is not None:
            ones.append(int(one))
    return np.mean(roots), np.mean(zeros), np.mean(ones)

def average_mica_overlap_by_type(ref_sents, hyp_sents, ref_stags, hyp_stags, ref_triples, hyp_triples, t='act-pass'):
    sents = 0
    roots = []
    zeros = []
    ones = []
    for ref_sent, hyp_sent, ref_stag, hyp_stag, refs, hyps in zip(ref_sents, hyp_sents, ref_stags, hyp_stags, ref_triples, hyp_triples):
        if t == 'act-pass':
            if not contains_modal(ref_sent) and not contains_modal(hyp_sent):
                if (contains_active(ref_stag) and contains_passive(hyp_stag)) or (contains_active(hyp_stag) and contains_passive(ref_stag)):
                    sents += 1
                    root, zero, one = mica_overlap(refs, hyps)
                    if root is not None:
                        roots.append(int(root))
                    if zero is not None:
                        zeros.append(int(zero))
                    if one is not None:
                        ones.append(int(one))
        elif t == 'modal':
            if contains_modal(ref_sent) or contains_modal(hyp_sent):
                sents += 1
                root, zero, one = mica_overlap(refs, hyps)
                if root is not None:
                    roots.append(int(root))
                if zero is not None:
                    zeros.append(int(zero))
                if one is not None:
                    ones.append(int(one))
        elif t == 'act-act':
            if not contains_modal(ref_sent) and not contains_modal(hyp_sent):
                if (contains_active(ref_stag) and contains_active(hyp_stag)) or (contains_passive(ref_stag) and contains_passive(hyp_stag)):
                    sents += 1
                    root, zero, one = mica_overlap(refs, hyps)
                    if root is not None:
                        roots.append(int(root))
                    if zero is not None:
                        zeros.append(int(zero))
                    if one is not None:
                        ones.append(int(one))
    return sents, np.mean(roots), np.mean(zeros), np.mean(ones)

def is_synonym(hyp, ref):
    if hyp == ref:
        return True
    if ref is None:
        return None
    for syn_pair in synonyms:
        if ref in syn_pair and hyp in syn_pair:
            return True
    return False

def get_average_mica_exclude_modals(ref_sents, hyp_sents, ref_mica, hyp_mica):
    precisions = []
    recalls = []
    fscores = []

    for ref_sent, hyp_sent, ref, hyp in zip(ref_sents, hyp_sents, ref_mica, hyp_mica):
        is_modal = False
        for word in ref_sent.split(' '):
            if word in modals:
                is_modal = True
        for word in hyp_sent.split(' '):
            if word in modals:
                is_modal = True
        if not is_modal:
            p, r, f = get_weighted_mica_score(ref, hyp)
            if p is not None and r is not None and f is not None:
                precisions.append(p)
                recalls.append(r)
                fscores.append(f)
    return np.mean(precisions), np.mean(recalls), np.mean(fscores)

def contains_modal(sent):
    for word in sent.split(' '):
        if word in modals:
            return True
    return False

def contains_active(stags):
    return 't27' in stags or 't83' in stags

def contains_passive(stags):
    return 't331' in stags or 't1252' in stags 

directions = ['uni', 'bi']
sizes = ['50', '100', '256']
types = ['linear', 'hierarchical']


reference_sents = read_seqs(REFERENCE_DIR + 'test-ref-ordered-words.txt')
reference_mica = get_mica_triples(REFERENCE_DIR + 'test-ref-ordered-mica-output.txt')
reference_stags = read_seqs(REFERENCE_DIR + 'test-ref-ordered-supertags.txt')

gold_sents = read_seqs(REFERENCE_DIR + 'test-para-ordered-words.txt')
gold_stags = read_seqs(REFERENCE_DIR + 'test-para-ordered-supertags.txt')
gold_mica = get_mica_triples(REFERENCE_DIR + 'test-para-ordered-mica-output.txt')

print('Gold Paraphrases')
roots, zeros, ones = average_mica_overlap(reference_mica, gold_mica)
print('For all {} pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f} overlap:'.format(len(gold_sents), roots, zeros, ones))
num_sents, roots, zeros, ones = average_mica_overlap_by_type(reference_sents, gold_sents, reference_stags, gold_stags, reference_mica, gold_mica, 'act-pass')
print('For {} active-passive pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f}'.format(num_sents, roots, zeros, ones))
num_sents, roots, zeros, ones = average_mica_overlap_by_type(reference_sents, gold_sents, reference_stags, gold_stags, reference_mica, gold_mica, 'act-act')
print('For {} active-active pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f}'.format(num_sents, roots, zeros, ones))
rouge_p, rouge_r, rouge_f = get_average_rouge(reference_sents, gold_sents)
print('Average ROUGE {:.3f}, {:.3f}, {:.3f}'.format(rouge_p, rouge_r, rouge_f))

for d in directions:
    for s in sizes:
        for t in types:
            output_loc = 'linear-hierarchical-experiment/model-outputs/' + d + 'directional-' + s +'/'

            pre = 'test-' + d + '-' + s + '-' 
            if t == 'linear':
                output_sents = read_seqs(output_loc + pre + t + '-output.txt')
            else:
                output_sents = read_seqs(output_loc + pre + t + '-output-ordered.txt')

            output_mica = get_mica_triples(output_loc + pre + t + '-mica-output.txt')

            mica_p, mica_r, mica_f = get_average_mica(reference_mica, output_mica)
            rouge_p, rouge_r, rouge_f = get_average_rouge(reference_sents, output_sents)
            print(t, d, s)
            roots, zeros, ones = average_mica_overlap(reference_mica, output_mica)
            print('For all {} pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f} overlap:'.format(len(output_sents), roots, zeros, ones))
            num_sents, roots, zeros, ones = average_mica_overlap_by_type(reference_sents, gold_sents, reference_stags, gold_stags, reference_mica, output_mica, 'act-pass')
            print('For {} active-active and active-passive pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f}'.format(num_sents, roots, zeros, ones))
            num_sents, roots, zeros, ones = average_mica_overlap_by_type(reference_sents, gold_sents, reference_stags, gold_stags, reference_mica, output_mica, 'modal')
            print('For {} modal pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f}'.format(num_sents, roots, zeros, ones))
            num_sents, roots, zeros, ones = average_mica_overlap_by_type(reference_sents, gold_sents, reference_stags, gold_stags, reference_mica, output_mica, 'act-act')
            print('For {} active-active pairs, overlap Root {:.3f}, Subject {:.3f}, Object {:.3f}'.format(num_sents, roots, zeros, ones))
            print('Average ROUGE {:.3f}, {:.3f}, {:.3f}'.format(rouge_p, rouge_r, rouge_f))

            
            
