#!/bin/bash

if [ "$1" == png ];
	then 	
		mkdir -p png
		for i in out/*.dat;
		do 
			k=${i##*/}
			j="png/${k%.dat}"
			gnuplot -e "inputfilename='${i}'; outputfilename='${j}'" 3DJulia2png;
		done
fi
if [ "$1" == eps ]:
	then 	
		mkdir -p eps
		for i in out/*.dat;
		do 
			k=${i##*/}
			j="eps/${k%.dat}"
			gnuplot -e "inputfilename='${i}'; outputfilename='${j}'" 3DJulia2eps;
		done
fi
