### openNMT

This directory contains code to train an openNMT seq2seq model on new data. More information [here](https://github.com/OpenNMT/OpenNMT-py#quickstart). 

The basic steps in training and running a model are done from the command line. First, preprocess the data:

`onmt_preprocess -train_src data/src-train.txt -train_tgt data/tgt-train.txt -valid_src data/src-val.txt -valid_tgt data/tgt-val.txt -save_data data/demo`

Next, train the model:
`onmt_train -data data/demo -save_model demo-model`

Finally, run the model on test data:
`onmt_translate -model demo-model_acc_XX.XX_ppl_XXX.XX_eX.pt -src data/src-test.txt -output pred.txt -replace_unk -verbose`

