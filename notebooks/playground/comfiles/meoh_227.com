%nproc=4
%mem=5760MB
%chk=meoh_227.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4215 0.1027 -0.0232
C 0.0268 0.0085 -0.0002
H 1.7783 -0.7565 0.2851
H -0.3233 -0.7047 -0.7464
H -0.3301 -0.4097 0.9409
H -0.5496 0.9239 -0.1341

