%nproc=4
%mem=5760MB
%chk=meoh_936.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4428 0.0981 -0.0661
C -0.0125 -0.0271 0.0106
H 1.6333 -0.3773 0.7695
H -0.3649 0.9719 -0.2461
H -0.2843 -0.8282 -0.6768
H -0.2780 -0.2151 1.0509

