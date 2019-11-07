import numpy as np

with open('phrase-table-filtered-0.6-joint-count.txt', 'r') as f:
    shared_tags = []
    for line in f:
        l = line.split('|||')
        s1 = set(l[0].strip().split(' '))
        s2 = set(l[1].strip().split(' '))
        shared = 0
        for tag in s1:
            if tag in s2:
                shared += 1
        shared_tags.append(shared/len(s1))

    print('Average shared tags between s1 and s2:', np.mean(shared_tags))
        