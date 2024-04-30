%nproc=4
%mem=5760MB
%chk=meoh_646.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4231 -0.0030 -0.0185
C 0.0185 -0.0043 0.0146
H 1.7615 0.9101 0.0921
H -0.3865 -1.0074 -0.1192
H -0.3664 0.4041 0.9490
H -0.3585 0.5776 -0.8264

