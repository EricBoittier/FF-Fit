%nproc=4
%mem=5760MB
%chk=meoh_454.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4444 0.0538 -0.0724
C -0.0153 -0.0137 0.0116
H 1.6805 0.0750 0.8786
H -0.2496 0.3192 1.0227
H -0.3575 0.7264 -0.7119
H -0.3582 -1.0258 -0.2033

