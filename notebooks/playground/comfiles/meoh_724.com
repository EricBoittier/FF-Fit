%nproc=4
%mem=5760MB
%chk=meoh_724.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4247 0.0683 0.0494
C 0.0054 -0.0122 -0.0006
H 1.8151 -0.1079 -0.8321
H -0.3425 -0.8711 -0.5745
H -0.3122 -0.0855 1.0395
H -0.3793 0.9102 -0.4356

