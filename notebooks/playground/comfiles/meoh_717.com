%nproc=4
%mem=5760MB
%chk=meoh_717.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4381 0.0582 0.0529
C -0.0103 -0.0110 0.0033
H 1.7373 0.0393 -0.8802
H -0.3107 -0.8952 -0.5589
H -0.3382 -0.0482 1.0422
H -0.3329 0.8965 -0.5070

