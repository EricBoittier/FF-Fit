%nproc=4
%mem=5760MB
%chk=meoh_445.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4014 0.0817 -0.0438
C 0.0399 -0.0026 -0.0033
H 1.8965 -0.3975 0.6532
H -0.3274 0.4117 0.9356
H -0.4771 0.5221 -0.8068
H -0.3527 -1.0170 -0.0728

