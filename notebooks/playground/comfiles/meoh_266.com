%nproc=4
%mem=5760MB
%chk=meoh_266.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 0.0359 -0.0581
C 0.0079 0.0000 0.0006
H 1.7897 0.2843 0.8176
H -0.4096 -0.3388 -0.9476
H -0.2758 -0.7188 0.7693
H -0.3790 0.9880 0.2499

