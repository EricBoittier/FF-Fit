%nproc=4
%mem=5760MB
%chk=meoh_539.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4328 0.1094 0.0120
C 0.0058 -0.0003 0.0100
H 1.7046 -0.7712 -0.3214
H -0.2400 -0.8148 0.6914
H -0.5193 0.9109 0.2966
H -0.2973 -0.2734 -1.0008

