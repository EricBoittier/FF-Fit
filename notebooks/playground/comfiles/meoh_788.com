%nproc=4
%mem=5760MB
%chk=meoh_788.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4337 0.1156 -0.0288
C -0.0027 -0.0194 -0.0001
H 1.7576 -0.7344 0.3360
H -0.4034 -0.5476 -0.8653
H -0.2278 -0.5356 0.9331
H -0.3916 0.9984 0.0293

