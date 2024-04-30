%nproc=4
%mem=5760MB
%chk=meoh_816.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4157 0.0677 -0.0652
C 0.0330 -0.0110 0.0105
H 1.7915 -0.1237 0.8195
H -0.4557 -0.2765 -0.9270
H -0.3255 -0.7156 0.7609
H -0.4165 0.9573 0.2306

