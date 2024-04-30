%nproc=4
%mem=5760MB
%chk=meoh_356.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4247 0.0980 -0.0502
C 0.0338 -0.0083 0.0049
H 1.6935 -0.5960 0.5874
H -0.4817 0.8261 -0.4706
H -0.3959 -0.9155 -0.4198
H -0.3733 0.0133 1.0158

