%nproc=4
%mem=5760MB
%chk=meoh_669.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4227 -0.0013 0.0126
C 0.0144 0.0034 0.0151
H 1.7566 0.8303 -0.3841
H -0.3092 -0.9977 -0.2702
H -0.3909 0.2433 0.9981
H -0.3513 0.6897 -0.7487

