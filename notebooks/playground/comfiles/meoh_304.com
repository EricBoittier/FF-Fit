%nproc=4
%mem=5760MB
%chk=meoh_304.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4482 -0.0043 0.0394
C -0.0164 0.0051 -0.0119
H 1.6186 0.7924 -0.5053
H -0.3220 0.1839 -1.0428
H -0.2924 -0.9776 0.3705
H -0.3368 0.7940 0.6686

