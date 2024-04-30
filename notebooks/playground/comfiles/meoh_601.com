%nproc=4
%mem=5760MB
%chk=meoh_601.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4189 0.0388 -0.0664
C 0.0250 0.0061 0.0103
H 1.7947 0.2350 0.8172
H -0.3005 -1.0127 0.2205
H -0.3901 0.6288 0.8028
H -0.4642 0.2458 -0.9338

