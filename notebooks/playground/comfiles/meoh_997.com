%nproc=4
%mem=5760MB
%chk=meoh_997.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4186 0.1012 0.0332
C 0.0119 -0.0199 0.0026
H 1.8247 -0.5341 -0.5929
H -0.3740 0.7035 0.7208
H -0.3811 0.2881 -0.9663
H -0.2687 -1.0420 0.2567

