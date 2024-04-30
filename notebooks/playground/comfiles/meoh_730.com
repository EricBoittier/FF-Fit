%nproc=4
%mem=5760MB
%chk=meoh_730.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4192 0.0757 0.0469
C 0.0297 -0.0063 -0.0008
H 1.7702 -0.2454 -0.8100
H -0.3374 -0.8569 -0.5752
H -0.3999 -0.1372 0.9923
H -0.4542 0.8994 -0.3662

