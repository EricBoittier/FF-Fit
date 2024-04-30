%nproc=4
%mem=5760MB
%chk=meoh_225.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4196 0.1017 -0.0181
C 0.0207 0.0100 -0.0016
H 1.8156 -0.7616 0.2237
H -0.3002 -0.7260 -0.7388
H -0.3083 -0.3993 0.9536
H -0.5275 0.9391 -0.1575

