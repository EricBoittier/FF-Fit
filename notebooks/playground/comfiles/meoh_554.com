%nproc=4
%mem=5760MB
%chk=meoh_554.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4136 0.1147 -0.0217
C 0.0333 -0.0172 0.0056
H 1.8254 -0.7527 0.1745
H -0.3780 -0.8424 0.5870
H -0.3882 0.8765 0.4657
H -0.4340 -0.1125 -0.9745

