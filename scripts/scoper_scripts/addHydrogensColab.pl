#!/usr/bin/perl -w

use strict;
use FindBin;
my $home = "$FindBin::Bin";

if ($#ARGV<0) {
  print "Usage: addHydrogens.pl <pdb1> <pdb2> ...\n";
  print "Adds hydrogens to first MODEL only!!!\n";
  exit;
}

for(my $i=0; $i<=$#ARGV; $i++) {
  my $file = $ARGV[$i];
  if(-e $file) {
    `reduce -OH -HIS -NOADJust -Xplor $file > $file.HB`;
  } else {
    print "File not found $file\n";
  }
}
