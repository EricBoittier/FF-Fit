%nproc=4
%mem=5760MB
%chk=meoh_903.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4174 0.0941 0.0382
C 0.0377 -0.0132 0.0146
H 1.7271 -0.4343 -0.7268
H -0.3815 0.6147 -0.7716
H -0.3476 -1.0019 -0.2345
H -0.4868 0.2702 0.9271

