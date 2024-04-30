%nproc=4
%mem=5760MB
%chk=meoh_628.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4386 0.0039 -0.0426
C -0.0050 -0.0032 0.0049
H 1.7101 0.7804 0.4902
H -0.3516 -1.0356 0.0512
H -0.2590 0.5215 0.9259
H -0.4150 0.4944 -0.8739

