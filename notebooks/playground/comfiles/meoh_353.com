%nproc=4
%mem=5760MB
%chk=meoh_353.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4276 0.1080 -0.0398
C 0.0091 -0.0179 0.0066
H 1.7433 -0.6910 0.4318
H -0.3609 0.8469 -0.5441
H -0.3552 -0.9638 -0.3941
H -0.3358 0.0919 1.0348

