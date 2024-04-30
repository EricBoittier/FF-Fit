%nproc=4
%mem=5760MB
%chk=meoh_456.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4535 0.0461 -0.0768
C -0.0221 -0.0119 0.0155
H 1.6113 0.1785 0.8814
H -0.2692 0.2827 1.0354
H -0.3449 0.7589 -0.6843
H -0.3467 -1.0249 -0.2225

