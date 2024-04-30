%nproc=4
%mem=5760MB
%chk=meoh_436.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4511 0.1138 -0.0340
C -0.0181 -0.0057 0.0130
H 1.5958 -0.7954 0.3022
H -0.4264 0.5201 0.8761
H -0.3855 0.4067 -0.9267
H -0.1427 -1.0845 0.1077

