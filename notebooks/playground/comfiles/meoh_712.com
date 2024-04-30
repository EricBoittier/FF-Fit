%nproc=4
%mem=5760MB
%chk=meoh_712.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4324 0.0478 0.0512
C 0.0163 -0.0045 0.0082
H 1.6821 0.1435 -0.8917
H -0.3115 -0.9032 -0.5143
H -0.4563 -0.0224 0.9902
H -0.3852 0.8609 -0.5192

