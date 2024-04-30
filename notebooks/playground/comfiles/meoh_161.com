%nproc=4
%mem=5760MB
%chk=meoh_161.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4259 0.0395 0.0634
C 0.0272 0.0087 -0.0128
H 1.7189 0.1507 -0.8653
H -0.3265 -0.9615 -0.3617
H -0.4152 0.1113 0.9781
H -0.4750 0.7537 -0.6299

