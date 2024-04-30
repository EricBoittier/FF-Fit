%nproc=4
%mem=5760MB
%chk=meoh_7.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4300 -0.0102 -0.0002
C 0.0103 0.0024 0.0001
H 1.7234 0.9249 0.0002
H -0.3661 -1.0206 -0.0001
H -0.3604 0.5138 0.8884
H -0.3604 0.5136 -0.8884

