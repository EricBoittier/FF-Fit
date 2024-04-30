%nproc=4
%mem=5760MB
%chk=meoh_538.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4309 0.1089 0.0144
C 0.0134 0.0005 0.0090
H 1.6966 -0.7600 -0.3528
H -0.2619 -0.8031 0.6921
H -0.5320 0.9019 0.2884
H -0.3161 -0.2887 -0.9890

