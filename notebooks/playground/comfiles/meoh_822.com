%nproc=4
%mem=5760MB
%chk=meoh_822.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4278 0.0539 -0.0620
C 0.0095 -0.0073 -0.0060
H 1.7778 0.0639 0.8534
H -0.4896 -0.2594 -0.9417
H -0.2519 -0.7401 0.7574
H -0.3545 0.9520 0.3617

