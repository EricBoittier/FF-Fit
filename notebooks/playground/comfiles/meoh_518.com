%nproc=4
%mem=5760MB
%chk=meoh_518.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4295 0.0757 0.0476
C -0.0032 -0.0181 0.0066
H 1.7879 -0.2042 -0.8205
H -0.3368 -0.5938 0.8700
H -0.3399 1.0176 0.0520
H -0.3037 -0.4201 -0.9610

