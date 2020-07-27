# Unused Files

This directory contains data and scripts that were not ultimately used in the work presented in the thesis paper.

## ParaNMT Data

Training, validation, and test sets were randomly selected from the [ParaNMT-5M dataset](https://github.com/jwieting/para-nmt-50m). The basic paraphrasing model described in the paper was used to generate test outputs. The poor quality of model outputs led to a refinement of model architecture and generation of artificial data which was ultimately used in the thesis paper.

## phrase-table

This directory contains code and phrase tables from training a supertag translation model using [Moses](http://www.cs.cmu.edu/afs/cs/project/cmt-55/lti/Courses/731/homework/mosesdecoder/scripts/moses-for-mere-mortals/). The full phrase table is included as `phrase-table.paraphrase.for_train.ref-para.gz`, and two filtered-down tables are included as `phrase-table-filtered-0.6-counts.txt` and `phrase-table-filtered-0.6-joint-count.txt`. `prune_phrase_table.py` can be used to filter lines from a phrase table according to custom conditions, and `phrase_table_stats.py` can be used to retrieve general statistics and information about a phrase table.