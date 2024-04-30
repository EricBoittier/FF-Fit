%nproc=4
%mem=5760MB
%chk=meoh_623.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4252 0.0151 -0.0457
C 0.0204 -0.0140 0.0003
H 1.7645 0.6969 0.5712
H -0.4408 -0.9971 0.0946
H -0.2726 0.5494 0.8862
H -0.4569 0.4626 -0.8559

