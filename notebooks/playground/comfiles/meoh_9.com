%nproc=4
%mem=5760MB
%chk=meoh_9.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4328 -0.0107 -0.0002
C 0.0074 0.0022 0.0001
H 1.7229 0.9255 0.0002
H -0.3664 -1.0217 -0.0000
H -0.3599 0.5145 0.8893
H -0.3599 0.5142 -0.8893

