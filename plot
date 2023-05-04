#!/usr/bin/gnuplot -p

plot 'accuracy.log' u 1:2 title "val" w lines, 'accuracy.log' u 1:3 title "train" w lines