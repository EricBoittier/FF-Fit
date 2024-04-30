%nproc=4
%mem=5760MB
%chk=meoh_432.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4488 0.1162 -0.0209
C -0.0127 -0.0026 0.0159
H 1.6169 -0.8441 0.0790
H -0.4570 0.5669 0.8322
H -0.3809 0.3206 -0.9578
H -0.1654 -1.0716 0.1640

