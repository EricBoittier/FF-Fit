%nproc=4
%mem=5760MB
%chk=meoh_572.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4203 0.0874 -0.0541
C 0.0116 0.0065 0.0135
H 1.8236 -0.4988 0.6199
H -0.2095 -0.9734 0.4367
H -0.4541 0.7847 0.6182
H -0.3829 0.0072 -1.0026

