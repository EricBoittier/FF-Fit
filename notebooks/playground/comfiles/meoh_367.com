%nproc=4
%mem=5760MB
%chk=meoh_367.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4375 0.0457 -0.0641
C -0.0205 0.0162 0.0072
H 1.7628 -0.0004 0.8592
H -0.4753 0.9397 -0.3514
H -0.2161 -0.8280 -0.6539
H -0.1846 -0.2449 1.0526

