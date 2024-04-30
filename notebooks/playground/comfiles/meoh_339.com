%nproc=4
%mem=5760MB
%chk=meoh_339.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4169 0.1002 0.0260
C 0.0332 0.0045 0.0015
H 1.7572 -0.7118 -0.4046
H -0.4562 0.5910 -0.7762
H -0.2854 -1.0144 -0.2184
H -0.4419 0.2751 0.9444

