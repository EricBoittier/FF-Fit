%nproc=4
%mem=5760MB
%chk=meoh_226.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4202 0.1021 -0.0206
C 0.0245 0.0096 -0.0010
H 1.7981 -0.7594 0.2544
H -0.3123 -0.7158 -0.7416
H -0.3202 -0.4047 0.9464
H -0.5411 0.9302 -0.1452

