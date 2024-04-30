%nproc=4
%mem=5760MB
%chk=meoh_906.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4195 0.1054 0.0321
C 0.0434 -0.0213 0.0160
H 1.6730 -0.5597 -0.6416
H -0.3852 0.6697 -0.7099
H -0.3537 -0.9921 -0.2809
H -0.5234 0.2480 0.9072

