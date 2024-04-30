%nproc=4
%mem=5760MB
%chk=meoh_307.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4476 0.0010 0.0466
C -0.0036 0.0019 -0.0093
H 1.5570 0.7117 -0.6193
H -0.3213 0.2438 -1.0236
H -0.3460 -0.9799 0.3178
H -0.3659 0.7718 0.6719

