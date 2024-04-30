%nproc=4
%mem=5760MB
%chk=meoh_237.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4402 0.1058 -0.0433
C -0.0102 -0.0171 0.0073
H 1.7125 -0.6246 0.5509
H -0.3300 -0.5943 -0.8603
H -0.2823 -0.4760 0.9578
H -0.3798 1.0051 -0.0741

