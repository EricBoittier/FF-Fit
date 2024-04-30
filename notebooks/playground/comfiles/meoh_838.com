%nproc=4
%mem=5760MB
%chk=meoh_838.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4145 0.0198 -0.0623
C 0.0382 -0.0047 0.0253
H 1.7533 0.5473 0.6910
H -0.3790 -0.0642 -0.9799
H -0.3813 -0.8583 0.5577
H -0.4419 0.9019 0.3938

