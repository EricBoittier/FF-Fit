%nproc=4
%mem=5760MB
%chk=meoh_391.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4487 -0.0115 -0.0027
C -0.0265 -0.0029 0.0072
H 1.6645 0.9377 -0.1163
H -0.2891 1.0475 0.1326
H -0.3293 -0.3798 -0.9697
H -0.2660 -0.6031 0.8850

