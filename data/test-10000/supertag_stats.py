input_file = 'moses-pred.txt'
mica_file = 'mica-parser-error-moses.txt'

seq_lengths = []
with open(input_file, 'r') as f:
    for line in f:
        seq_lengths.append(len(line.strip().split()))

parser_results = []
with open(mica_file, 'r') as f:
    for line in f:
        if "Earley Parser (TRUE)" in line:
            parser_results.append(True)
        elif "Earley Parser (FALSE)" in line:
            parser_results.append(False)

# print(len(seq_lengths))
# print(len(parser_results))
good_parse_lengths = []
bad_parse_lengths = []
for i in range(len(seq_lengths)):
    if parser_results[i] == True:
        good_parse_lengths.append(seq_lengths[i])
    else:
        bad_parse_lengths.append(seq_lengths[i])

print('Average sequence length of supertags in {}: {}'.format(input_file, sum(seq_lengths) * 1.0/len(seq_lengths)))
print('Average sequence length of good parses: {}'.format(sum(good_parse_lengths)*1.0/len(good_parse_lengths)))
print('Average sequence length of bad parses: {}'.format(sum(bad_parse_lengths) * 1.0/len(bad_parse_lengths)))

## BLSTM STAGGER RESULTS
# Average sequence length of supertags in test-para-supertags.txt: 10.5425
# Average sequence length of good parses: 9.43673401789
# Average sequence length of bad parses: 12.2276627966

## MOSES RESULTS
# Average sequence length of supertags in moses-pred.txt: 7.4502
# Average sequence length of good parses: 7.2234648231
# Average sequence length of bad parses: 7.66961826053

## OPENNMT RESULTS
# Average sequence length of supertags in opennmt-pred.txt: 9.6744
# Average sequence length of good parses: 9.38222903102
# Average sequence length of bad parses: 10.7305029995