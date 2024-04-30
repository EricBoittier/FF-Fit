%nproc=4
%mem=5760MB
%chk=meoh_455.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4497 0.0500 -0.0750
C -0.0199 -0.0130 0.0137
H 1.6426 0.1269 0.8828
H -0.2576 0.3019 1.0298
H -0.3487 0.7441 -0.6981
H -0.3515 -1.0263 -0.2129

