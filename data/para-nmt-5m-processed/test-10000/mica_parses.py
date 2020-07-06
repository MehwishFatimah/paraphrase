num_correct = 0
num_incorrect = 0
input_file = 'mica-parser-error-opennmt.txt'
with open(input_file, 'r') as f:
    for line in f:
        if "Earley Parser (TRUE)" in line:
            num_correct += 1
        elif "Earley Parser (FALSE)" in line:
            num_incorrect += 1

print('MICA Parser Results for {}'.format(input_file))
print('Total Successful Parses: {} ({}%)'.format(num_correct, num_correct * 100.0/(num_incorrect + num_correct)))
print('Total Failed Parses: {} ({}%)'.format(num_incorrect, num_incorrect * 100.0/(num_incorrect + num_correct)))
print('Total Parses: {}'.format(num_correct + num_incorrect))

## BLSTM-STAGGER RESULTS ON PARAPHRASES
# MICA Parser Results for mica-parser-error-para.txt
# Total Successful Parses: 6038 (60.38%)
# Total Failed Parses: 3962 (39.62%)

## MOSES TRANLSATED PARAPHRASES
# MICA Parser Results for mica-parser-error-moses.txt
# Total Successful Parses: 4918 (49.18%)
# Total Failed Parses: 5082 (50.82%)

## OPENNMT TRANSLATED PARAPHRASES
# MICA Parser Results for mica-parser-error-opennmt.txt
# Total Successful Parses: 7833 (78.33%)
# Total Failed Parses: 2167 (21.67%)
