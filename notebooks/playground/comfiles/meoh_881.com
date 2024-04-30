%nproc=4
%mem=5760MB
%chk=meoh_881.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4261 0.0195 0.0416
C 0.0354 -0.0088 0.0080
H 1.6506 0.5697 -0.7378
H -0.3644 0.4134 -0.9140
H -0.4352 -0.9886 0.0899
H -0.4503 0.5862 0.7814

