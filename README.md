# Syntactically-Controlled Paraphrase Generation with Tree-Adjoining Grammar

This repository contains code and trained models associated with my senior thesis work in Cognitive Science at Yale University. The thesis document is located [here](https://cogsci.yale.edu/sites/default/files/files/2020ThesisWIDDER.pdf).
With any questions or concerns, please contact the author Sarah Widder at sarah.j.widder@gmail.com.

## Package Requirements
I recommend two separate conda environments. One environment will use Python 3 and PyTorch for running the model and the various utility scripts. The other will use Python 2 and TensorFlow 1.0 to run the pretrained supertagger.
### paraphrase environment
- nltk 3.4.5
- numpy 1.17.2
- python 3.7.4
- pytorch

### supertagger environment
- tensorflow 1.14.0
- python 2.7.16
- numpy 1.16.5


## Training and Testing the Model

This step by step guide will walk you through training, testing, and evaluating the model using new data. 
We will assume your data is located at `data/pairs.txt`, with one pair of paraphrases per line, reference and paraphrase sentences separated by tabs.

1. First, prepare the data by splitting the pairs into training and test sets, and separating the reference and paraphrase sentences into separate files, as well as generating part of speech sequences for each sentence. Make sure the path at the top of `prepare_data.py` in `scripts` is correct, then call the following at the command line:
    > `(paraphrase)$ python scripts/prepare_data.py`

    This assumes you're in a conda environment called `paraphrase` with the packages listed above. You only need `nltk` for this part. You should now have eight additional files in the `data` directory that separately contain the training and test sets of the reference and paraphrase sentences and their corresponding sequences of part of speech tags. 

2. Next we will generate supertag sequences for the data. Navigate to the `bilstm_stagging` directory.
    > `(paraphrase)$ cd bilstm_stagging`
    
    Activate your supertagging environment, or just make sure you're using Python 2 and Tensorflow 1.0.

    > `(paraphrase)$ conda activate supertagging`

    Before using the pre-trained supertagger for the first time, you'll need to download the pre-trained model from this [link](https://drive.google.com/drive/folders/1CzL7i0jnGT9BhQkM8vmiR-JbRohjIBZI?usp=sharing) and place it in `/bilstm_stagging/tag_pretrained/Pretrained_Stagger/`. See the `bilstm_stagging` README for more info.

    Repeat the following steps four times (once for each set of training and test data, reference and paraphrase):

    1. Copy and paste the contents of your sentence data (e.g. `data/train-ref-words.txt`) into `bilstm_stagging/tag_pretrained/sents/test.txt`. 
    2. Copy and paste the contents of the corresponding POS data (e.g. `data/train-ref-tags.txt`) into `bilstm_stagging/tag_pretrained/predicted_pos/test.txt`.
    3. Run the pretrained supertagger:
    > `(supertagging)$ python scripts/run_pretrained.py tag_pretrained/config_pretrained.json tag_pretrained/Pretrained_Stagger/best_model --no_gold`
    4. Copy and paste the contents of the supertagger output in `bilstm_stagging/tag_pretrained/predicted_stag/test.txt` to a new file (e.g. `data/train-ref-supertags.txt`) in the `data` directory.
  
    This is unfortunately pretty time consuming. Note that you may need to comment out some lines relating to `elmo` in the scripts to prevent errors.

3. You should now have three files for each set of training reference, training paraphrase, testing reference, and testing paraphrase data. The three files for each set have the corresponding sentences, part of speech sequences, and supertag sequences, one per line, in the separate files. Now we'll generate [MICA](https://mica.lis-lab.fr/) parses for each set of data.

    Note - MICA is a 32-bit binary, so it will not work on Mac OS 10.15 or later. Clone this repo to Grace and it should run there.

    1. First return to the root `paraphrase` directory and your paraphrase conda environment, then format your data for MICA.

        > `(supertagging)$ conda activate paraphrase`
        > `(paraphrase)$ cd ..`
        > `(paraphrase)$ python scripts/mica_format.py`

    This should generate four additional files in your `data` folder named like `train-ref-mica-input.txt`.

    2. Set an environment variable for the `mica-1_0.x86_32` directory: 
        > `(paraphrase)$ cd mica-1_0.x86_32`
        > `(paraphrase)$ export MicaRoot=$PWD`
        > `(paraphrase)$ cd ..`
    3. Run the MICA script for each set of data:
        > `(paraphrase)$ perl $MicaRoot/scripts/mica.perl -PR -i train-ref-mica-input.txt > train-ref-mica-output.txt`
        > `(paraphrase)$ perl $MicaRoot/scripts/mica.perl -PR -i train-para-mica-input.txt > train-para-mica-output.txt`
        > `(paraphrase)$ perl $MicaRoot/scripts/mica.perl -PR -i test-ref-mica-input.txt > test-ref-mica-output.txt`
        > `(paraphrase)$ perl $MicaRoot/scripts/mica.perl -PR -i test-para-mica-input.txt > test-para-mica-output.txt`

      The `-PR` option runs the parser and post-processor which formats the output nicely. The output is explained in the MICA README.

4. Now we have the words, part of speech tags, supertags, and MICA parses for each sentence in our training and test data. If you need do not need to order any data hierarchically, skip this step. Modify the `PREFIX` at the top of `hierarchical_format.py` in `scripts` to `train` or `test` as appropriate, then run:
    > `(paraphrase)$ python scripts/hierarchical_format.py`

    You should now have new sets of training and testing data. Since not all sentences may have a good enough parse for reordering, the whole set (reference and paraphrase) has been filtered down to only those pairs for which reordering the paraphrase was possible. 

5. We're ready to train the model. Make sure that the parameters at the top of `model.py` in `model` are correct. You can modify the hidden size, the directionality of the supertag encoder, the number of iterations for training, and of course the training, testing, and save paths. Train the model:
    > `(paraphrase)$ python model/model.py`

    This should be done on Grace or another GPU machine.

6. Once the model has been trained, the last checkpoint will be stored at the save path specified in `model.py`. Make sure that the parameters at the top of `test.py` in `model` match those used during training, and then test the model using the last checkpoint:
    > `(paraphrase)$ python model/test.py`

    This should also be done on Grace or another GPU machine.

7. The model output on the test data should now be stored in `data` with the rest of your data, in the format `test-<direction>-<order>-<hidden size>-output.txt`. So, for a bidirectional model on linearly ordered data with hidden size 100, the model output will be named `test-bidir-lin-100-output.txt`.

    We'll need to get supertag sequences for the model output. Make sure the paths in `tag_sents.py` are correct, then run:
    > `(paraphrase)$ python scripts/tag_sents.py`

    Now follow step 2 above, copying and pasting your output words and tags into the appropriate files in `bilstm_stagging` and running the pretrained supertagger. 

8. Generate MICA parses for the model output by changing the path in `mica_format.py` appropriately and repeating step 3 above.

9. There are two kinds of model evaluation you can now run. These are described in detail in the thesis paper.
    > `(paraphrase)$ python scripts/model_evaluation.py` 
    
    This will run statistical analysis on the word-for-word and supertag-for-supertag accuracy of the output compared to the gold paraphrases in your test set. You may need to modify the paths at the bottom of the file appropriately for your model outputs.

    > `(paraphrase)$ python scripts/score.py`

    This will score the model outputs based on deep syntactic roles from the MICA parse and ROUGE scores. Again, you may need to modify the paths at the bottom of the file.

