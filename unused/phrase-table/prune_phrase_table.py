phrase_table_list = []
with open('phrase-table-filtered-0.6-counts.txt', 'r') as f:
    with open('phrase-table-filtered-0.6-joint-count.txt', 'w') as w:
        # filter phrase table with only most likely transitions

        for line in f:
            l = line.split('|||')
            
            counts = l[4].strip().split(' ')
            joint_count = counts[2]

            if int(joint_count) > 1:
                w.write(line)

