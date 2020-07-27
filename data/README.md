# Data

The `data` directory contains all training, testing, and model output data. 

## Artificial Data

Three sets of artificial data, with associated model outputs, are included.

* Set 1
  This was the first set generated using the lexicalized context-free grammar. It had several issues, and was not ultimately used in the final paper.
* Set 2
  This set was generated using the second version of the lexicalized context-free grammar in `grammar.py`. More detail on sentence generation can be found in the final paper. 
* linear-hierarchical experiment
  These sets were generated using the same `grammar.py` but include the linear and hierarchical orderings used in the paper. Model outputs for each ordering (linear vs. hierarchical) and three hidden sizes (50, 100, or 256) are included.
* Set 3
  These sets were generated using the third version of the grammar in `new_grammar.py`. This grammar allows more complexity in sentence construction and paraphrase relations, and excludes some lexical items and constructions that were problematic from set 2. Model output for a linear ordering with hidden size 100 is included.