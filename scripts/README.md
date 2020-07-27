# Python scripts

This directory contains scripts used to generate data, format data, and evaluate model outputs. 


- `grammar.py` This file contains the CFG used to generate artificial dataset 2 as well as the linear and hierarchically-ordered data used in the final paper.
- `hierarchical_format.py` This file performs the hierarchical (top-down, left-to-right) ordering using MICA parser outputs.
- `mica_format.py` This file formats text, part of speech, and supertag data for input into the MICA parser.
- `model_evaluation.py` This script performs statistical analysis of model outputs, such as word-for-word and supertag-for-supertag accuracy.
- `new_grammar.py` This file contains the third CFG which builds off of the CFG in `grammar.py`, adding more complex paraphrase operations and resolving some issues. 
- `retrieve_order.py` takes hierarchical output data and attempts to reconstruct a linear sentence using the original linear data.
- `score.py` implements a variety of more complicated model evaluation functions, including calculating the ROUGE scores and variations on parse overlap. The metrics used in the final paper are described there in more detail.
- `prepare_data.py` splits data into training and test sets, separates reference and paraphrase sentences, and uses the NLTK part of speech tagger to assign part of speech tags to text.
- `tag_sents.py` assigns part of speech tags to each word in a set of sentences.
- `word_freq.pkl` contains a dictionary dump of word frequencies calculated from the Brown corpus. This allows the frequency of synonyms to be taken into account when doing synonym substitution.