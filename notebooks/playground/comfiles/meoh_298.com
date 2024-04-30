%nproc=4
%mem=5760MB
%chk=meoh_298.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4228 -0.0024 0.0155
C 0.0070 0.0033 -0.0059
H 1.8338 0.8550 -0.2220
H -0.3736 0.0537 -1.0261
H -0.3095 -0.9442 0.4303
H -0.3582 0.8197 0.6172

