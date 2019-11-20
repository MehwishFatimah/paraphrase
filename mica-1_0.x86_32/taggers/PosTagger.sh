#!/bin/tcsh
set ngram=3
set context=6
set p=$MicaRoot/taggers/utils/
set pos=$MicaRoot/taggers/
set model=$MicaRoot/taggers/models

rm -f .posinput .posoutput >& /dev/null
cat $1 |\
sed -e 's/[ ]*$//g' |\
perl $p/pitoify | grep -v '^$' |tee .posinput |\
perl $pos/posformatter $context |\
$p/MaxEntDecoder -m $model/pos.mdl -d $model/pos.mdl/featdict.pos -n 1 |\
awk -F':' '{print "/"$1}' > .posoutput
paste -d'/' .posinput .posoutput 
