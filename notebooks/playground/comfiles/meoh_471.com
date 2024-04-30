%nproc=4
%mem=5760MB
%chk=meoh_471.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4115 0.0025 -0.0306
C 0.0222 0.0050 0.0061
H 1.8562 0.7701 0.3861
H -0.3077 -0.0146 1.0448
H -0.4359 0.8271 -0.5438
H -0.3227 -0.8971 -0.4991

