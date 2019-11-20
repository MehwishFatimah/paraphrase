#!/usr/bin/perl

# This program takes an input file and makes the output so punctuation
# symbols and contractions are separated into their own words.
#
# It also replaces any underscores with dashes because the collins
# parser seems to puke when presented with underscores.

$BRACKET    = "\Q(\E|\Q)\E|\Q[\E|\Q]\E";
$MULTI_PU   = "\Q--\E|\Q...\E|\Q``\E|\Q''\E";
$SINGLE_PU1 = "\Q!\E|\Q'\E|\Q-\E|\Q`\E";
$SINGLE_PU2 = "\Q,\E|\Q.\E|\Q:\E|\Q;\E|\Q?\E|\"";
$CONTRACTS  = "\Q'd\E|\Q'm\E|\Q'll\E|\Q'em\E|\Q're\E|\Q's\E|\Q've\E|\Qn't\E|\Q'D\E|\Q'M\E|\Q'LL\E|\Q'EM\E|\Q'RE\E|\Q'S\E|\Q'VE\E|\QN'T\E";

select STDIN; $| = 1;
select STDOUT; $| = 1;

while (<>) {
  chomp;

  $line = $_;
  $line =~ s/_/-/g;
  $prevwd_p   = 0;
  $prevwd_len = 0;

  while ($line !~ /^$/) {
    if ($line =~ /^([ \t]*)($MULTI_PU)(.*)$/) {
      $pu   = $2;
      $line = $3;

      print "$pu ";
    }
    elsif ($line =~ /^([ \t]*)($CONTRACTS)([^A-Za-z].*)$/) {
      $cntr = $2;
      $line = $3;

      print "$cntr ";
    }
    elsif ($line =~ /^([ \t]*)($SINGLE_PU1)([^A-Za-z].*)$/ && $prevwd_p == 0) {
      $pu   = $2;
      $line = $3;

      print "$pu ";
    }
    elsif ($line =~ /^([ \t]*)($SINGLE_PU2)(.*)$/) {
      $pu   = $2;
      $line = $3;

      print &single_pu12str($pu, $prevwd_p), " ";
    }
    elsif ($line =~ /^([ \t]*)($BRACKET)(.*)$/) {
      $brak = $2;
      $line = $3;

      print &bracket2str($brak), " ";
    }
    elsif ($line =~ /^([ \t]*)([A-Za-z][A-Za-z]?\.)( |$SINGLE_PU2)(.*)$/) {
      # Mr., Dr., Co., etc.
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([A-Z]\.[A-Z])([^A-Za-z].*|)$/) {
      # N.Y, N.J, ...
      $wd = $2;
      $line = $3;
      if ($line =~ /^\.([^A-Za-z]|)(.*)$/) {
        $line = "$1$2";
        $wd  .= ".";
      }

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([0-9])([0-9.,-]*)([0-9])([^A-Za-z0-9].*|)$/) {
      # phone no's (555-1212), scores (6-2), large no's (1,024)
      $wd = "$2$3$4";
      $line = $5;

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t()-.`',":;!?\[\]]+?)($MULTI_PU)(.*)$/) {
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t()-.`',":;!?\[\]]+?)($CONTRACTS)([^A-Za-z].*|)$/){
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t()-.`',":;!?\[\]]+?)($SINGLE_PU1)([^A-Za-z].*|)$/
           && $prevwd_p == 0) {
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t()-.`',":;!?\[\]]+?)($SINGLE_PU2)(.*)$/) {
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t()-.`',":;!?\[\]]+?)($BRACKET)(.*)$/) {
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t]+)($SINGLE_PU2)(.*)$/) {
      $wd   = $2;
      $line = "$3$4";

      print "$wd ";
    }
    elsif ($line =~ /^([ \t]*)([^ \t]+)(.*)$/) {
      $wd   = $2;
      $line = $3;

      print "$wd ";
    }
    elsif ($line =~ /^[ \t]*$/) {
      $line = "";
    }

    if ($line =~ /^ /) {
      $prevwd_p = 0;
    }
    else {
      $prevwd_p = 1;
    }
  }

  print "\n";
}

sub single_pu12str {
  my ($pu, $prevwd_p) = @_;
  my $outstr;

  if ($pu =~ /^\"/) {
    if ($prevwd_p == 0) {
      $outstr = "``";
    }
    else {
      $outstr = "''";
    }
  }
  else {
    $outstr = $pu;
  }

  return $outstr;
}

sub bracket2str {
  my ($brak) = @_;
  my $outstr;

  if ($brak =~ /\Q(\E/) {
    $outstr = "-LCB-";
  }
  elsif ($brak =~ /\Q)\E/) {
    $outstr = "-RCB-";
  }
  if ($brak =~ /\Q[\E/) {
    $outstr = "-LRB-";
  }
  elsif ($brak =~ /\Q]\E/) {
    $outstr = "-RRB-";
  }

  return $outstr;
}

