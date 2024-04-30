%nproc=4
%mem=5760MB
%chk=meoh_459.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4540 0.0340 -0.0774
C -0.0138 -0.0065 0.0180
H 1.5738 0.3296 0.8493
H -0.3176 0.2159 1.0409
H -0.3640 0.7846 -0.6450
H -0.3466 -1.0077 -0.2559

