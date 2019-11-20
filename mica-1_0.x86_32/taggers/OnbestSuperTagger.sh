#!/bin/tcsh
set MicaRoot=/home/srini/SuperTagger/Distrib/
set ngram=3
set context=6
set nbest=1
set p=$MicaRoot/bin/utils/
set stag=$MicaRoot/bin/stag/
set model=$MicaRoot/models
set PosTagger=$stag/../pos/PosTagger.sh
rm -f .staginput .stagoutput >& /dev/null
 # remove the scores from the POStagger

cat $1 |\
sed -e 's/:/_COLON_/g'|\
$PosTagger |\
awk -F':' '{print $1}' |\
tee .staginput |\
perl $stag/stagformatter $context |\
tee .stagdata |\
awk -F',' '{for (i=1; i<=NF; i++) printf("%s_%d ", $i, i); printf "\n";}'|\
perl $p/gram.pl $model/stag.mdl/featdict.stags $ngram |\
$p/MaxEntDecoder -m $model/stag.mdl -n $nbest |\
awk '{print "/"$0}' > .stagoutput
sed -e 's/_COLON_/:/g' .staginput |\
paste -d'/' - .stagoutput

