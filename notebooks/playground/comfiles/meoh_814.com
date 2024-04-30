%nproc=4
%mem=5760MB
%chk=meoh_814.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4187 0.0723 -0.0680
C 0.0263 -0.0112 0.0183
H 1.7770 -0.1851 0.8072
H -0.4051 -0.2974 -0.9409
H -0.3305 -0.7164 0.7690
H -0.4142 0.9691 0.2001

