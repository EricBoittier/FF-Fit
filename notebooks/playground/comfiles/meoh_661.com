%nproc=4
%mem=5760MB
%chk=meoh_661.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4294 -0.0061 0.0082
C 0.0217 -0.0038 -0.0018
H 1.6864 0.9091 -0.2302
H -0.3992 -0.9969 -0.1592
H -0.3594 0.3171 0.9677
H -0.4148 0.6972 -0.7132

