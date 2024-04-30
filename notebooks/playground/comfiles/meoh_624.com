%nproc=4
%mem=5760MB
%chk=meoh_624.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4276 0.0132 -0.0447
C 0.0158 -0.0122 0.0005
H 1.7565 0.7146 0.5556
H -0.4292 -1.0036 0.0857
H -0.2646 0.5436 0.8952
H -0.4527 0.4673 -0.8589

