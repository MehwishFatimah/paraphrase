''' Generates hierarchically-ordered data from a MICA parse. '''
DATA_DIR = 'data/'
PREFIX = 'train'

def read_line(line, index):
    l, attributes = line.strip().split('||')
    l = l.split(' ')
    attributes = [att for att in attributes.split(' ') if att]

    line_info = {
        'index': index,
        'ID': l[0],
        'word': l[1],
        'POS': l[2],
        'parentID': l[3],
        'parentWord': l[4],
        'parentPOS': l[5],
        'supertag': l[6],
        'parentSupertag': l[7],
        'difference': l[8]
    }
    for item in attributes:
        att, val = item.split(':')
        line_info[att] = val

    return line_info

def process_sent(lines, sent_number):

    line_infos = []
    for i in range(len(lines)):
        line_infos.append(read_line(lines[i], i))
    supertags, indices, words = get_full_seq(line_infos)
    if len(supertags) < len(lines):
        # print('Skipping...', [item['word'] for item in line_infos])
        skips.add(sent_number)
        return [],[],[]
    return supertags, indices, words

def get_full_seq(line_infos):
    supertags = []
    indices = []
    words = []
    root = None

    for i in range(len(line_infos)):
        line_info = line_infos[i]
        if line_info['DRole'] == 'Root':
            root = line_info

    if root:
        supertags.append(root['supertag'])
        indices.append(str(root['index']))
        words.append(root['word'])

        left_lines = line_infos[:root['index']]
        right_lines = line_infos[root['index'] + 1:]

        left_tags, left_indices, left_words = get_seq(root, left_lines)
        right_tags, right_indices, right_words = get_seq(root, right_lines)

        supertags.extend(left_tags)
        supertags.extend(right_tags)

        indices.extend(left_indices)
        indices.extend(right_indices)

        words.extend(left_words)
        words.extend(right_words)

    return supertags, indices, words


def get_seq(parent, lines):
    supertags = []
    indices = []
    words = []
    if lines:
        # get queue of things with this as the parent
        queue = []
        for line in lines:
            if line['parentID'] == parent['ID']:
                queue.append(line)
        
        # for each thing in the queue, look left and right
        while queue:
            nxt = queue.pop(0)
            supertags.append(nxt['supertag'])
            indices.append(str(nxt['index']))
            words.append(nxt['word'])

            left_lines = lines[:nxt['index']]
            right_lines = lines[nxt['index'] + 1:]

            left_tags, left_indices, left_words = get_seq(nxt, left_lines)
            right_tags, right_indices, right_words = get_seq(nxt, right_lines)

            supertags.extend(left_tags)
            supertags.extend(right_tags)

            indices.extend(left_indices)
            indices.extend(right_indices)

            words.extend(left_words)
            words.extend(right_words)

    return supertags, indices, words

def get_seq_by_attribute(lines, attribute):
    line_infos = []
    for i in range(len(lines)):
        line_infos.append(read_line(lines[i], i))
    return [item[attribute] for item in line_infos]


ref_supertags = []
ref_indices = []
ref_words = []
ref_original_order_words = []
ref_original_order_supertags = []

para_supertags = []
para_indices = []
para_words = []
para_original_order_words = []
para_original_order_supertags = []

skips = set()

with open(DATA_DIR + PREFIX + '-ref-mica-output.txt', 'r') as input_file:
    
    sent_num = 0
    sent_lines = []
    line = input_file.readline()
    while line != '':
        if line[0] == '#':
            supertags, indices, words = process_sent(sent_lines, sent_num)
            original_seq = get_seq_by_attribute(sent_lines, 'word')
            ref_original_order_words.append(' '.join(original_seq) + '\n')
            ref_original_order_supertags.append(' '.join(get_seq_by_attribute(sent_lines, 'supertag')) + '\n')

            ref_supertags.append(' '.join(supertags) + '\n')
            ref_indices.append(' '.join(indices) + '\n')
            ref_words.append(' '.join(words) + '\n')
            
            sent_num += 1
            sent_lines = []
            line = input_file.readline()
            line = input_file.readline()
            line = input_file.readline()
            line = input_file.readline()
            line = input_file.readline()
        else:
            sent_lines.append(line)
            line = input_file.readline()
    print(sent_num + 1)

with open(DATA_DIR + PREFIX + '-para-mica-output.txt', 'r') as input_file:
    
                sent_num = 0
                sent_lines = []
                line = input_file.readline()
                while line != '':
                    if line[0] == '#':
                        supertags, indices, words = process_sent(sent_lines, sent_num)
                        original_seq = get_seq_by_attribute(sent_lines, 'word')
                        para_original_order_words.append(' '.join(original_seq) + '\n')

                        para_original_order_supertags.append(' '.join(get_seq_by_attribute(sent_lines, 'supertag')) + '\n')

                        para_supertags.append(' '.join(supertags) + '\n')
                        para_indices.append(' '.join(indices) + '\n')
                        para_words.append(' '.join(words) + '\n')
                        
                        sent_num += 1
                        sent_lines = []
                        line = input_file.readline()
                        line = input_file.readline()
                        line = input_file.readline()
                        line = input_file.readline()
                        line = input_file.readline()
                    else:
                        sent_lines.append(line)
                        line = input_file.readline()
                print(sent_num + 1)

print(len(ref_supertags), len(para_supertags))
print(len(skips))

# with open(DATA_DIR + PREFIX + '-ref-reordered-supertags.txt', 'w') as supertag_file:
#     with open(DATA_DIR + PREFIX + '-ref-reordered-indices.txt', 'w') as index_file:
#         with open(DATA_DIR + PREFIX + '-ref-reordered-words.txt', 'w') as word_file:
with open(DATA_DIR + PREFIX + '-ref-ordered-supertags.txt', 'w') as ordered_supertags:
    with open(DATA_DIR + PREFIX + '-ref-ordered-words.txt', 'w') as ordered_file:
        for i in range(len(ref_supertags)):
            if i not in skips:
                # supertag_file.write(ref_supertags[i])
                # index_file.write(ref_indices[i])
                # word_file.write(ref_words[i])
                ordered_supertags.write(ref_original_order_supertags[i])
                ordered_file.write(ref_original_order_words[i])
                    
with open(DATA_DIR + PREFIX + '-para-reordered-supertags.txt', 'w') as supertag_file:
        with open(DATA_DIR + PREFIX + '-para-reordered-indices.txt', 'w') as index_file:
            with open(DATA_DIR + PREFIX + '-para-reordered-words.txt', 'w') as word_file:
                with open(DATA_DIR + PREFIX + '-para-ordered-words.txt', 'w') as ordered_words:
                    with open(DATA_DIR + PREFIX + '-para-ordered-supertags.txt', 'w') as ordered_stags:
                        for i in range(len(para_supertags)):
                            if i not in skips:
                                supertag_file.write(para_supertags[i])
                                index_file.write(para_indices[i])
                                word_file.write(para_words[i])
                                ordered_words.write(para_original_order_words[i])
                                ordered_stags.write(para_original_order_supertags[i])