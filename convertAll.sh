#!/bin/bash

mkdir -p png

for i in out/*.dat;
do k=${i##*/}
j="png/${k%.dat}"
gnuplot -e "inputfilename='${i}'; outputfilename='${j}'" 3DJulia2png;
done
