%nproc=4
%mem=5760MB
%chk=meoh_721.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4326 0.0646 0.0514
C -0.0097 -0.0132 0.0009
H 1.7939 -0.0440 -0.8531
H -0.3252 -0.8824 -0.5762
H -0.2929 -0.0674 1.0521
H -0.3405 0.9103 -0.4742

