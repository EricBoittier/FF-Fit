%nproc=4
%mem=5760MB
%chk=meoh_599.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4177 0.0444 -0.0647
C 0.0282 0.0016 0.0054
H 1.8090 0.1831 0.8231
H -0.3224 -1.0012 0.2493
H -0.3781 0.6392 0.7905
H -0.4872 0.2402 -0.9250

