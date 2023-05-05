#!/usr/bin/gnuplot -p

while (1) {
	plot 'accuracy.log' u 1:2 title "val" w lines, 'accuracy.log' u 1:3 title "train" w lines
	pause 7
}
