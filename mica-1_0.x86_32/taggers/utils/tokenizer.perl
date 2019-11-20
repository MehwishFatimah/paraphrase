use PennTreebank::Tokenizer;

while (<>) {
    chomp;
    my @words = split;
    @tokenized_sents =  PennTreebank::Tokenizer::tokenize(@words);
    for $s (@tokenized_sents){
	print "$s ";
    }
    print "\n";
}
