#!/usr/bin/perl -w

use strict;


if ($#ARGV < 0) {
  print "plotFit.pl <fit_file1> <color> <legend> <fit_file2> <color> <legend> ... \n";
  print "The colors are: 1-green, 2-red, 3-blue, 4-light blue, 5-lighter blue, 6-grey-blue, 7-grey, 10-black\n";
  exit;
}

my $fitFile = $ARGV[0];
my $outFile = trimExtension($fitFile) . ".eps";

open OUT, ">gnuplot_fit.txt";
print OUT "set terminal postscript eps size 3.0,2.3 color enhanced  linewidth 2.5 font 'Helvetica,18';  set output '$outFile';\n";
print OUT "set encoding iso_8859_1;set xlabel 'q (\305^{-1})';set ylabel 'I(q) log-scale' offset 1;\n";
print OUT "set style line 10 lc rgb '#333333' lt 1 lw 1.0\n"; # black
print OUT "set style line 1 lc rgb '#1a9850' lt 1 lw 1.0\n"; # green
print OUT "set style line 2 lc rgb '#e26261' lt 1 lw 1.0\n"; # red
print OUT "set style line 3 lc rgb '#3288bd' lt 1 lw 1.0\n"; # blue      .
print OUT "set style line 4 lc rgb '#6baed6' lt 1 lw 1.0\n"; #      .
print OUT "set style line 5 lc rgb '#A6CEE3' lt 1 lw 1.0\n"; #      .
print OUT "set style line 6 lc rgb '#A1B1C1' lt 1 lw 1.0\n"; # grey-blue
print OUT "set style line 7 lc rgb '#909090' lt 1 lw 1.0\n"; # grey
print OUT "set style line 8 lc rgb '#606060' lt 1 lw 1.0\n"; # darker grey
print OUT "set key spacing 1.3\n";
print OUT "set key font \",17\"\n";
print OUT "set style line 11 lc rgb '#808080' lt 1; set border 3 back ls 11;set xtics nomirror;set ytics nomirror;set format y '';\n";
print OUT "set lmargin 2; set rmargin 2;set multiplot\n";
print OUT "set origin 0.04,0;set size 0.96,0.3; set tmargin 0; set bmargin 3.3;set ylabel '';set format y '';set xtics nomirror;set ytics nomirror; set border 3\n";
print OUT "set style line 11 lc rgb '#808080' lt 1;set border 3 back ls 11;f(x)=0\n";

my $residuals = "plot f(x) notitle lc rgb '#333333'";
my $plots = "plot '$fitFile' u 1:2 notitle lc rgb '#333333' pt 6 ps 0.8";

for(my $i=0; $i<$#ARGV; $i+=3) {
  $fitFile = $ARGV[$i];
  my $chi = `grep Chi $fitFile | awk '{print \$11}'`;
  chomp $chi;
  $chi = sprintf("%.2f", $chi);
  my $color = $ARGV[$i+1];
  my $caption = $ARGV[$i+2];
  print "$fitFile $chi $color\n";

  $residuals .= ", '$fitFile' u 1:((\$2-\$3)/\$4) notitle w l ls $color";
  $plots .= ", '$fitFile' u 1:3 t '$caption {/Symbol c}^2 = $chi' w l ls $color";
}

print OUT "$residuals\n";
print OUT "set origin 0.04,0.3;set size 0.96,0.7; set bmargin 0; set tmargin 1;set xlabel ''; set format x ''; set ylabel 'I(q) log-scale';set log y;\n";
print OUT "$plots\n";
print OUT "unset multiplot;reset\n";
close OUT;

`gnuplot gnuplot_fit.txt`;

sub trimExtension {
  my $str = shift;
  $str =~ s/\.[^.]+$//;
  return $str;
}
