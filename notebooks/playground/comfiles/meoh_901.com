%nproc=4
%mem=5760MB
%chk=meoh_901.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4202 0.0858 0.0419
C 0.0223 -0.0079 0.0118
H 1.7671 -0.3412 -0.7692
H -0.3657 0.5881 -0.8143
H -0.3161 -1.0216 -0.2024
H -0.4345 0.2920 0.9550

