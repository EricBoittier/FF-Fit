%nproc=4
%mem=5760MB
%chk=meoh_583.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4298 0.0794 -0.0693
C 0.0078 -0.0146 0.0157
H 1.7325 -0.2415 0.8059
H -0.3134 -0.9922 0.3754
H -0.3444 0.7712 0.6840
H -0.4043 0.1608 -0.9780

