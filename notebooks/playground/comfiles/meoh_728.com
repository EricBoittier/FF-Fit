%nproc=4
%mem=5760MB
%chk=meoh_728.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4182 0.0730 0.0474
C 0.0264 -0.0083 -0.0014
H 1.7966 -0.1983 -0.8150
H -0.3496 -0.8591 -0.5696
H -0.3743 -0.1176 1.0064
H -0.4378 0.9010 -0.3833

