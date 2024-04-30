%nproc=4
%mem=5760MB
%chk=meoh_431.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4438 0.1152 -0.0169
C -0.0059 -0.0009 0.0159
H 1.6486 -0.8425 0.0202
H -0.4629 0.5782 0.8184
H -0.3870 0.2961 -0.9612
H -0.1864 -1.0638 0.1763

