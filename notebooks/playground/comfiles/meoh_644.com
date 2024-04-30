%nproc=4
%mem=5760MB
%chk=meoh_644.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4181 -0.0036 -0.0233
C 0.0279 -0.0014 0.0174
H 1.7663 0.8983 0.1375
H -0.3751 -1.0071 -0.1025
H -0.4052 0.4074 0.9302
H -0.3693 0.5611 -0.8275

