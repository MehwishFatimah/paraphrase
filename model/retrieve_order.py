OUTPUT_DIR = 'linear-hierarchical-experiment/model-outputs/bidirectional-50/500k-iter/'
DATA_DIR = 'linear-hierarchical-experiment/test/'
PREFIX = 'test'

indices_list = []
with open(DATA_DIR + PREFIX + '-para-reordered-indices.txt', 'r') as ind:
    for line in ind:
        indices_list.append(line.strip().split(' '))

output_words = []
with open(OUTPUT_DIR + PREFIX + '-bi-50-hierarchical-output.txt', 'r') as out:
    for line in out:
        output_words.append(line.strip().split(' '))

with open(OUTPUT_DIR + PREFIX + '-bi-50-hierarchical-output-ordered.txt', 'w') as ordered:
    for i in range(len(indices_list)):
        indices = indices_list[i]
        output = output_words[i]

        ordered_words = [''] * len(indices)
        for j in range(min(len(output), len(indices))):
            ordered_words[int(indices[j])] = output[j]

        ordered_words = [w for w in ordered_words if w]
        ordered.write(' '.join(ordered_words) + '\n')

