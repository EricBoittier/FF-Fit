%nproc=4
%mem=5760MB
%chk=meoh_948.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4088 0.0374 -0.0533
C 0.0362 -0.0075 0.0065
H 1.8535 0.3632 0.7570
H -0.4292 0.9663 -0.1459
H -0.4265 -0.5861 -0.7931
H -0.3324 -0.4617 0.9262

