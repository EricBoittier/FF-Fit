%nproc=4
%mem=5760MB
%chk=meoh_313.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4214 0.0226 0.0484
C 0.0368 -0.0100 0.0071
H 1.6704 0.4853 -0.7789
H -0.3227 0.3199 -0.9676
H -0.4618 -0.9636 0.1804
H -0.4269 0.7041 0.6876

