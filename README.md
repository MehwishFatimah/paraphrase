# Syntactically-Controlled Paraphrase Generation with Tree-Adjoining Grammar

This repository contains code and trained models associated with my senior thesis work in Cognitive Science at Yale University. The thesis document is located [here](https://cogsci.yale.edu/sites/default/files/files/2020ThesisWIDDER.pdf).
With any questions or concerns, please contact the author Sarah Widder at sarah.j.widder@gmail.com.

The below sections describe what can be found in each directory in the repo.

### data

The `data` directory contains all training, testing, and model output data. 

#### ParaNMT Data

Training, validation, and test sets were randomly selected from the [ParaNMT-5M dataset](https://github.com/jwieting/para-nmt-50m). The basic paraphrasing model described in the paper was used to generate test outputs. 

#### Artificial Data

Three sets of artificial data, with associated model outputs, are included.

* Set 1
  This was the first set generated using the lexicalized context-free grammar. It had several issues, and was not ultimately used in the final paper.
* Set 2
  This set was generated using the second version of the lexicalized context-free grammar in `grammar.py`. More detail on sentence generation can be found in the final paper. 
* linear-hierarchical experiment
  These sets were generated using the same `grammar.py` but include the linear and hierarchical orderings used in the paper. Model outputs for each ordering (linear vs. hierarchical) and three hidden sizes (50, 100, or 256) are included.
* Set 3
  These sets were generated using the third version of the grammar in `new_grammar.py`. This grammar allows more complexity in sentence construction and paraphrase relations, and excludes some lexical items and constructions that were problematic from set 2. Model output for a linear ordering with hidden size 100 is included.

### model 

The `model` directory contains trained model checkpoints and all original python scripts written for this work.

#### Model checkpoints

The last model checkpoint for encoder, decoder, and supertag encoder are included from each model run. The directory for each run generally indicates the model parameters used for that run.

#### Python scripts

- `attention.py` This file contains implementations of various attention mechanisms provided by Prof. Robert Frank.
- `grammar.py` This file contains the CFG used to generate artificial dataset 2 as well as the linear and hierarchically-ordered data used in the final paper.
- `input_format.py` This file performs the hierarchical (top-down, left-to-right) ordering using MICA parser outputs.
- `mica_format.py` This file formats text, part of speech, and supertag data for input into the MICA parser.
- `model_evaluation.py` This script performs statistical analysis of model outputs, such as word-for-word and supertag-for-supertag accuracy.
- `model_results.md` This file contains a description of model performances with various input orderings and model sizes.
- `model.py` This contains the actual PyTorch model including encoder, decoder, and supertag encoder, as well as functions for training and evaluation. *To train the model, simply change the parameter values at the top of the file to reflect the desired parameters and training data, then run* `python model.py`. 
- `new_grammar.py` This file contains the third CFG which builds off of the CFG in `grammar.py`, adding more complex paraphrase operations and resolving some issues. 
- `retrieve_order.py` takes hierarchical output data and attempts to reconstruct a linear sentence using the original linear data.
- `score.py` implements a variety of more complicated model evaluation functions, including calculating the ROUGE scores and variations on parse overlap. The metrics used in the final paper are described there in more detail.
- `tag_sents.py` uses the NLTK part of speech tagger to assign part of speech tags to text.
- `test.py` contains code to test a trained paraphrase model. *To test a model, change the parameter values at the top to match those used during training and the path to test data, then run* `python test.py`.
- `word_freq.pkl` contains a dictionary dump of word frequencies calculated from the Brown corpus. This allows the frequency of synonyms to be taken into account when doing synonym substitution.


### bilstm_stagging

This submodule contains scripts to run a pre-trained TAG supertagger on textual data. See the README in this directory for how to generate supertags for new data. Note: These scripts require Python 2. You may also need to comment out some lines relating to `elmo` in the scripts to prevent errors.

### MICA

This directory contains the pre-trained [MICA parser](https://mica.lis-lab.fr/) which takes in supertagged data and performs TAG parsing. To use the parser:

1. Prepare your data using `mica_format.py` in the `model` folder of the `paraphrase` repo. Make sure the filenames in the python script are correct - you need to have three parallel files:
    * text: one sentence per line, words separated by spaces
    * POS tags: one sentence per line, parts of speech separated by spaces
    * supertags: one sentence per line, supertags separated by spaces
  This will create a file containing your MICA-formatted input data.
2. Set an environment variable for the `mica-1_0.x86_32` directory: `cd mica-1_0.x86_32` then `export MicaRoot=$PWD`.
3. Navigate back to the MICA-formatted input data and run the MICA script: `perl $MicaRoot/scripts/mica.perl -PR -i mica-input.txt > mica-output.txt` using the appropriate input and output filenames. The `-PR` option runs the parser and post-processor which formats the output nicely. The output is explained in the MICA README.

### openNMT

This directory contains code to train an openNMT seq2seq model on new data. More information [here](https://github.com/OpenNMT/OpenNMT-py#quickstart). 

The basic steps in training and running a model are done from the command line. First, preprocess the data:

`onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo`

Next, train the model:
`onmt_train -data data/demo -save_model demo-model`

Finally, run the model on test data:
`onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose`

### phrase-table

This directory contains code and phrase tables from training a supertag translation model using [Moses](http://www.cs.cmu.edu/afs/cs/project/cmt-55/lti/Courses/731/homework/mosesdecoder/scripts/moses-for-mere-mortals/). The full phrase table is included as `phrase-table.paraphrase.for_train.ref-para.gz`, and two filtered-down tables are included as `phrase-table-filtered-0.6-counts.txt` and `phrase-table-filtered-0.6-joint-count.txt`. `prune_phrase_table.py` can be used to filter lines from a phrase table according to custom conditions, and `phrase_table_stats.py` can be used to retrieve general statistics and information about a phrase table.