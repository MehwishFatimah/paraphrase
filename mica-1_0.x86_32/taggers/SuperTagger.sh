#!/bin/tcsh
set ngram=3
set context=6
set nbest=100
set p=$MicaRoot/taggers/utils/
set stag=$MicaRoot/taggers/
set model=$MicaRoot/taggers/models
set PosTagger=$MicaRoot/taggers/PosTagger.sh
rm -f .staginput .stagoutput >& /dev/null
 # remove the scores from the POStagger

cat $1 |\
sed -e 's/:/_COLON_/g'|\
$PosTagger |\
awk -F':' '{print $1}' |\
tee .staginput |\
perl $stag/stagformatter $context |\
tee .stagdata |\
$p/MaxEntDecoder -m $model/stag.mdl -d $model/stag.mdl/featdict.stags -n $nbest |\
awk '{print "/"$0}' > .stagoutput
sed -e 's/_COLON_/:/g' .staginput |\
paste -d'/' - .stagoutput

