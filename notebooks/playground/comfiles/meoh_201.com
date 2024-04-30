%nproc=4
%mem=5760MB
%chk=meoh_201.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4281 0.1017 0.0261
C 0.0086 -0.0037 0.0010
H 1.7987 -0.6825 -0.4303
H -0.2801 -0.8404 -0.6352
H -0.3955 -0.1776 0.9982
H -0.4343 0.9149 -0.3839

