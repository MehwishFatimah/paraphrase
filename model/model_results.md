# Model Results
## Trained 2/19/20 on artificial dataset 2

- Train pairs: 232,972
- Test pairs: 58,292

- vocab size:
  - reference: 91
  - paraphrase: 98
  - supertags: 96 

- bidirectional supertag encoding

### Evaluation metrics
- Average word-for-word accuracy:  0.9161446865121416
- Average word-for-word accuracy allowing synonyms: 0.9271601939409363
- Average supertag-for-supertag accuracy:  0.9663730485987501
- Of correct supertags, average word accuracy:  0.9301150725149904
- Of correct supertags, average word accuracy allowing synonyms: 0.9413406771427869

#### Active-passive dataset
- 50,000 test sentences with only active->passive and passive->active
- no synonym substitution

- Average word-for-word accuracy:  0.9364721926962941
- Average word-for-word accuracy allowing synonyms: 0.9549590977911551
- Average supertag-for-supertag accuracy:  0.9611281459096911
- Of correct supertags, average word accuracy:  0.9442308710281571
- Of correct supertags, average word accuracy allowing synonyms: 0.9633083035798929

#### Modal dataset
- 50,000 test sentences with only modal substitution
- no synonym substitution

- Average word-for-word accuracy:  0.9201716635588172
- Average word-for-word accuracy allowing synonyms: 0.9388021037851989
- Average supertag-for-supertag accuracy:  0.9699499732489892
- Of correct supertags, average word accuracy:  0.9367977741891534
- Of correct supertags, average word accuracy allowing synonyms: 0.9559060119809547

#### Using openNMT supertags instead of gold supertags
- Average word-for-word accuracy:  0.7622137801146811
- Average word-for-word accuracy allowing synonyms: 0.7722877764976755
- Average supertag-for-supertag accuracy:  0.7962539262014336
- Of correct supertags, average word accuracy:  0.9159944819580134
- Of correct supertags, average word accuracy allowing synonyms: 0.9287662271815313

## Linear-Hierarchical Experiment
- training set: 171,452 training pairs of reference and paraphrase
  - subset of original training set due to parser failures

- test set: 42,727 test pairs of reference and paraphrase
  - subset of original test set due to parser failures

- supertag encoder is single-directional GRU
- all hidden sizes are 50 (vocab and supertag set size around 100)

### Evaluation

#### Unidirectional Linear model
- inputs to decoder
  - hidden representation and attention over reference sentence, in original linear order, encoded with single-directional GRU
  - paraphrase sentence supertags, in original linear order, encoded with single-direction GRU (one supertag per decoder output)
- outputs
  - paraphrase sentence in original linear order

Hidden size 50
- Average word-for-word accuracy:  0.7027499132366055
- Average word-for-word accuracy allowing synonyms: 0.7076095100149342
- Average supertag-for-supertag accuracy:  0.9618675448505907
- Of correct supertags, average word accuracy:  0.7095373813048099
- Of correct supertags, average word accuracy allowing synonyms: 0.7145209120201427

Hidden size 100
Average word-for-word accuracy:  0.732395846745962
Average word-for-word accuracy allowing synonyms: 0.7396181328983463
Average supertag-for-supertag accuracy:  0.9217848932232764
Of correct supertags, average word accuracy:  0.764664748988225
Of correct supertags, average word accuracy allowing synonyms: 0.7725587482742468

Hidden size 256
Average word-for-word accuracy:  0.953637719828409
Average word-for-word accuracy allowing synonyms: 0.9657067087045198
Average supertag-for-supertag accuracy:  0.9784252356450456
Of correct supertags, average word accuracy:  0.9561314430726066
Of correct supertags, average word accuracy allowing synonyms: 0.9682938125514567

#### Unidirectional Hierarchical model
- inputs to decoder
  - hidden representation and attention over reference sentence, in original linear order, encoded with single-directional GRU
  - paraphrase sentence supertags, in hierarchical order according to top-down left-to-right traversal of parse tree given by MICA using the supertag sequence, encoded with single-directional GRU (one supertag per decoder output)
- outputs
  - paraphrase sentence in hierarchical order as supertags above

Hidden size 50
- Average word-for-word accuracy:  0.7795326713040435
- Average word-for-word accuracy allowing synonyms: 0.786276074240164
Running POS tagger and supertagger with hierarchical order:
- Average supertag-for-supertag accuracy:  0.5205087056901244
- Of correct supertags, average word accuracy:  0.8020169643111819
- Of correct supertags, average word accuracy allowing synonyms: 0.806256397937424
Running POS tagger and supertagger with reconstructed order:
- Average supertag-for-supertag accuracy:  0.959766659301745
- Of correct supertags, average word accuracy:  0.7915703549615627
- Of correct supertags, average word accuracy allowing synonyms: 0.7985288921384822

Hidden size 100
Average word-for-word accuracy:  0.8623404474026382
Average word-for-word accuracy allowing synonyms: 0.8738174013457435
Average supertag-for-supertag accuracy:  0.9621251805563169
Of correct supertags, average word accuracy:  0.8748335733164867
Of correct supertags, average word accuracy allowing synonyms: 0.886562620130184

Hidden size 256
Average word-for-word accuracy:  0.8624188176823966
Average word-for-word accuracy allowing synonyms: 0.8723636446652598
Average supertag-for-supertag accuracy:  0.941709035331408
Of correct supertags, average word accuracy:  0.8829161531093284
Of correct supertags, average word accuracy allowing synonyms: 0.8932002833829257

#### Bidirectional linear upper bound?
Hidden size: 50, Number of iterations: 250,000
Average word-for-word accuracy:  0.7012849230110031
Average word-for-word accuracy allowing synonyms: 0.7102091725644228

Hidden size: 50, Number of iterations: 500,000
Average word-for-word accuracy:  0.7089889126944415
Average word-for-word accuracy allowing synonyms: 0.7131989555036855
Average supertag-for-supertag accuracy:  0.9380881482608568
Of correct supertags, average word accuracy:  0.7342774558447109
Of correct supertags, average word accuracy allowing synonyms: 0.7386262892869637

Hidden size: 100, Number of iterations: 250,000
Average word-for-word accuracy:  0.7948738435193787
Average word-for-word accuracy allowing synonyms: 0.8013560528491577
Average supertag-for-supertag accuracy:  0.9418240493842567
Of correct supertags, average word accuracy:  0.8158921078245877
Of correct supertags, average word accuracy allowing synonyms: 0.8226925562626505

Hidden size: 256, Number of iterations: 250,000
Average word-for-word accuracy:  0.9532935719651474
Average word-for-word accuracy allowing synonyms: 0.9710590645569025
Average supertag-for-supertag accuracy:  0.9778923663652481
Of correct supertags, average word accuracy:  0.9561699324955084
Of correct supertags, average word accuracy allowing synonyms: 0.9739803818192895

#### Bidirectional hierarchical upper bound?
Hidden size: 50, Number of iterations: 250,000
Average word-for-word accuracy:  0.42489034334428083
Average word-for-word accuracy allowing synonyms: 0.42657908330237243

Hidden size: 50, Number of iterations: 500,000
Average word-for-word accuracy:  0.5358725098847331
Average word-for-word accuracy allowing synonyms: 0.5400642522300287
Average supertag-for-supertag accuracy:  0.6499323541221612
Of correct supertags, average word accuracy:  0.696051677009271
Of correct supertags, average word accuracy allowing synonyms: 0.7018907440704955

Hidden size: 100, Number of iterations: 250,000
Average word-for-word accuracy:  0.7313618293757271
Average word-for-word accuracy allowing synonyms: 0.7380627967827106
Average supertag-for-supertag accuracy:  0.9285257092857967
Of correct supertags, average word accuracy:  0.753016628181195
Of correct supertags, average word accuracy allowing synonyms: 0.7599731398506109

Hidden size: 256, Number of iterations: 250,000
Average word-for-word accuracy:  0.8516195550953002
Average word-for-word accuracy allowing synonyms: 0.8643398419330777
Average supertag-for-supertag accuracy:  0.9535052630585886
Of correct supertags, average word accuracy:  0.8690426825253674
Of correct supertags, average word accuracy allowing synonyms: 0.8821502182268692