%nproc=4
%mem=5760MB
%chk=meoh_585.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4375 0.0783 -0.0710
C -0.0050 -0.0203 0.0144
H 1.7126 -0.1925 0.8299
H -0.3070 -1.0019 0.3798
H -0.3048 0.7759 0.6958
H -0.3997 0.2037 -0.9766

