%nproc=4
%mem=5760MB
%chk=meoh_878.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4313 0.0146 0.0327
C 0.0140 -0.0089 0.0149
H 1.6631 0.6704 -0.6578
H -0.2655 0.3635 -0.9707
H -0.4181 -1.0046 0.1151
H -0.4062 0.6294 0.7921

