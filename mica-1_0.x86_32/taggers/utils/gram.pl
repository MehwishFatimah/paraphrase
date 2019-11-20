#!/usr/common/perl5.6.1/bin/perl
#extracts n-grams from text and convert them into LLAMA bsparse format
#usage gram.pl dict n-gram create_dict_flag  worddata > gram_data

$EOS="\\.\\.\\.EOS\\.\\.\\.";
$fdict=shift;
$ngram=shift;
$fdata=shift;

$create_dict_flag=0;

if ($ngram>3) {
    die "Only up to 3-gram so far\n";
}

if ($create_dict_flag==0) {
    open(FDICT, "<$fdict") or die "Unable to read file $fdict: $!";
    while ($line=<FDICT>) {
        chomp $line;
        my @temp=split(/ +/,$line);
        my $idx=$temp[0];
        my $gram=$temp[1];
        for $i (2..$#temp) {
            $gram .= " $temp[$i]";
        }
        $dict{$gram}=$idx;
    } 
} else {
    open(FDICT, ">$fdict") or die "Unable to read file $fdict: $!";
    while ($list=<>) {
        
        chomp $list;
        @words=split(/ +/, $list);
        for $i (0..$#words) {
            $dict{$words[$i]}++;
            if ($i>=1&& $ngram>1) {
                $dict{$words[$i-1]." ".$words[$i]}++;
                if ($i>=2 && $ngram>2) {
                    $dict{$words[$i-2]." ".$words[$i-1]." ".$words[$i]}++;
                }            
            }            
        }
    }
    my @stf=sort keys %dict;
    $i=1;
    foreach $w (@stf) {
        #removing infrequent words helps
        if ($dict{$w}>2) {
            print FDICT "$i $w\n";
            $dict{$w}=$i;
            $i++;
        } else {
            $dict{$w}=0;
        }
    }
}



while ($line=<>) {
    chomp $line;
    if ($line=~/$EOS/){
	print "...EOS...\n";
	next;
    }

    $line =~ s/\s+/ /g;
    my @words =split(' ', $line);
    for $i (0..$#words) {
        $idx=$dict{$words[$i]};
        if ($idx>0) {print " $idx";}
        if ($i>=1 && $ngram>1) {
            $idx=$dict{$words[$i-1]." ".$words[$i]};
            if ($idx>0) {printf(" %s", $idx);}
            if ($i>=2 && $ngram>2) {
                $idx=$dict{$words[$i-2]." ".$words[$i-1]." ".$words[$i]};
                if ($idx>0) {printf(" %s", $idx);}
            }
        }
    }
    printf("\n");
}
