%nproc=4
%mem=5760MB
%chk=meoh_8.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4360 -0.0113 -0.0002
C 0.0046 0.0024 0.0001
H 1.7228 0.9258 0.0002
H -0.3664 -1.0226 -0.0000
H -0.3601 0.5149 0.8902
H -0.3601 0.5147 -0.8902

