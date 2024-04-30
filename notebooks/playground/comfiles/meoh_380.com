%nproc=4
%mem=5760MB
%chk=meoh_380.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4175 0.0072 -0.0435
C 0.0457 -0.0182 0.0052
H 1.6621 0.7617 0.5322
H -0.2883 1.0165 -0.0714
H -0.5105 -0.5389 -0.7743
H -0.4482 -0.3515 0.9179

