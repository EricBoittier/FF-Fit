%nproc=4
%mem=5760MB
%chk=meoh_448.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4055 0.0722 -0.0503
C 0.0319 -0.0075 -0.0023
H 1.8929 -0.2360 0.7422
H -0.2819 0.3867 0.9642
H -0.4552 0.5831 -0.7782
H -0.3865 -1.0061 -0.1278

