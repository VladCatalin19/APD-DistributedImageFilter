build: homework
homework: homework.c
	mpicc -o homework homework.c -lm -Wall
serial: homework
	mpirun -np 1 homework in/lenna_bw.pgm out.pgm emboss emboss
distrib: homework
	mpirun -np 4 homework in/lenna_bw.pgm out2.pgm emboss emboss
clean:
	rm -f homework
