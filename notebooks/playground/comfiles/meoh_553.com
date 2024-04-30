%nproc=4
%mem=5760MB
%chk=meoh_553.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4131 0.1139 -0.0196
C 0.0303 -0.0157 0.0063
H 1.8369 -0.7549 0.1419
H -0.3661 -0.8426 0.5956
H -0.3872 0.8848 0.4566
H -0.4156 -0.1229 -0.9825

