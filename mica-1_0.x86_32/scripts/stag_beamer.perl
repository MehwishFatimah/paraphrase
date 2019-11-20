
#16/10 modified by alexis to work as a filter : reads on stdin and writes on stdout

# Modified by Benoit to work with new syntx parser, and then by Owen
# to generate only the .parserinput file.


# Based on launch_parser.0.2.pl.

# All output is to standard output.

# Modified by Owen to fix tCO problem: tCO comes out odd from analyzer:

# 1 t1 2 t3
# 2 t3 3 t83
# 3 t83 3 t83
# 4 t3 3 t83
# 3 tCO 3 t83
# 6 t1 8 t3
# 7 t36 8 t3
# 8 t3 3 t83
# 9 t26 3 t83

# Also now produces output directly compatible with depeval-forest.perl.

# Also now allows for nb_stags and beam_width to be passed in as args.


use strict;


#my $stagoutput = shift;

$| = 1;

# If two extra arguments, take them to be the beam width and the max stag nb
# If a third one, take it as the scale_factor
# Otherwise, use default values

my $beam_width = shift || 310; # the scores of the supertags given as input to the parser must be greater or equal
                               # than the best score / beam_width
#my $nb_max_de_supertags = shift || 8; #maximum number of supertags allowed per word for sentences of length < $scale_factor
my $scale_factor = shift || 60; # for sentences of length > $scale_factor, the maximum number of supertags allowed per word is
                                # max(1, $nb_max_de_supertags / (length / $scale_factor))

#print STDERR "beam width = $beam_width\n";
#print STDERR "max supertags = $nb_max_de_supertags\n";
#print STDERR "scale factor = $scale_factor\n";


my $VERBOSE = 0;


my $ff_pos;
my $ff;
my $pos;
my $stag;

my $nb_phrases = 0;
my $nb_stags = 0;
my $long_phrase = 0;

my @sentence_lines;

my $moy_stags_mot;

my @nb_mots_avec_n_stags = 0;  # not really used?


#open(CORPUS, $corpus_file) or die "Failure opening $corpus_file\n";
#open (PHRASE, ">$nom_fichier_phrase") or die "Failure opening $nom_fichier_phrase\n";


my $end_of_sentence = 0;

while (<STDIN>) {
  my $l_corpus = $_;
  if ($l_corpus =~ /\.\.\.EOS\.\.\./) {
    $end_of_sentence = 1;

    $nb_phrases++;
	
    if ($long_phrase ==0 ) {
      print "Empty sentence found: sentence number $nb_phrases has length  $long_phrase!\n";
    } else {
      $moy_stags_mot = $nb_stags / $long_phrase;
    }
  } else {
    if ($end_of_sentence == 1) {
      process_sentence ();
      @sentence_lines = ();
      $end_of_sentence = 0;
      $long_phrase = 0;
      $nb_stags = 0;
    }

    $long_phrase++;

    chop $l_corpus;

    push (@sentence_lines, $l_corpus);

	
  }    
}
process_sentence ();


sub process_sentence {
  my $sdag = "";
  my $nb_max_local_de_supertags;
  for my $l_corpus (@sentence_lines) {
    my @tab_corpus = split / /, $l_corpus;
    #	unshift @tab_corpus, "WORD";

    my $nb_supertags_mot = (@tab_corpus - 1) / 2;
    $ff_pos = $tab_corpus[0];
    
    ($ff, $pos) = split /\/\//, $ff_pos;
    my $max_prob = $tab_corpus[2];

#    print STDERR "********* max prob = $max_prob\n";
	
    my $tau = $max_prob / $beam_width;


#    $tab_mots[$long_phrase] = $ff_pos;

    $sdag .= "\"$ff $pos\" (";
    if ($VERBOSE) {
      print STDERR "$ff $pos (";
    }

    if($long_phrase > 150){
	$nb_max_local_de_supertags = 1;
    }
    else{
    $nb_max_local_de_supertags = $scale_factor / $long_phrase ** (1/3);
    if ($nb_max_local_de_supertags < 1) {$nb_max_local_de_supertags = 1}
    else {$nb_max_local_de_supertags = int($nb_max_local_de_supertags)}
}
#    $nb_max_local_de_supertags = $nb_max_de_supertags;
#    if ($scale_factor > 0 && $long_phrase > $scale_factor) {
#      $nb_max_local_de_supertags /= $long_phrase / $scale_factor;
#      if ($nb_max_local_de_supertags < 1) {$nb_max_local_de_supertags = 1}
#      else {$nb_max_local_de_supertags = int($nb_max_local_de_supertags)}
#    }

    my $nb_max_de_supertags_depasse = 0;
    my $proba_trop_faible = 0;
    for (my $i=1; (! $nb_max_de_supertags_depasse && ! $proba_trop_faible && ($i <= $nb_supertags_mot)); $i++) {
      $stag = @tab_corpus[2*$i - 1];
      my $proba = @tab_corpus[2*$i];

      if ($i > $nb_max_local_de_supertags) {
	$nb_max_de_supertags_depasse = 1;
	$nb_mots_avec_n_stags[$i-1]++;
      } elsif ($proba < $tau) {
	$proba_trop_faible = 1;
	$nb_mots_avec_n_stags[$i-1]++;
      } else {
	if ($VERBOSE) {
	  print STDERR "$stag [|$proba|] ";
	}
	$nb_stags++;
	$sdag .= "$stag [|$proba|] ";
      }
    }
    if ($VERBOSE) {
      print STDERR ")\n";
    }
    $sdag .= ")\n";
  }
  print "##SDAG BEGIN /* sent_id=$nb_phrases length=$long_phrase trans_nb=$nb_stags max_lexical_ambiguity=$nb_max_local_de_supertags */\n";
  print $sdag;
  print "##SDAG END\n";
}
