#!/bin/bash

for i in out/*.dat;
do gnuplot -e "inputfilename='${i}'; outputfilename='${i%.dat}'" 3DJulia2png;
done
