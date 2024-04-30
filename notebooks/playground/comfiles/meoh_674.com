%nproc=4
%mem=5760MB
%chk=meoh_674.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4350 0.0009 0.0188
C -0.0102 0.0035 0.0179
H 1.7341 0.7887 -0.4815
H -0.2640 -0.9995 -0.3254
H -0.3677 0.2234 1.0238
H -0.2993 0.7164 -0.7543

