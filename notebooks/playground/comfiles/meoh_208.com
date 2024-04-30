%nproc=4
%mem=5760MB
%chk=meoh_208.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4244 0.1164 0.0147
C 0.0282 -0.0212 0.0040
H 1.7254 -0.7734 -0.2649
H -0.3688 -0.7609 -0.6913
H -0.4349 -0.1985 0.9746
H -0.4081 0.9224 -0.3236

