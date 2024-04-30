%nproc=4
%mem=5760MB
%chk=meoh_765.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4384 0.1150 0.0111
C -0.0089 -0.0102 0.0037
H 1.7435 -0.7773 -0.2557
H -0.3021 -0.7037 -0.7845
H -0.2625 -0.4080 0.9863
H -0.4453 0.9694 -0.1915

