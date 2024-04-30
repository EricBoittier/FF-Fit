%nproc=4
%mem=5760MB
%chk=meoh_394.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4381 -0.0069 0.0082
C -0.0193 0.0058 0.0097
H 1.7593 0.8652 -0.3027
H -0.3497 1.0282 0.1931
H -0.2845 -0.3788 -0.9751
H -0.2620 -0.6883 0.8143

