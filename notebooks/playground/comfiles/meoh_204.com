%nproc=4
%mem=5760MB
%chk=meoh_204.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4243 0.1090 0.0211
C 0.0258 -0.0122 0.0030
H 1.7606 -0.7277 -0.3629
H -0.3403 -0.8007 -0.6545
H -0.4411 -0.1809 0.9734
H -0.4360 0.9093 -0.3514

