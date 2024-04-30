%nproc=4
%mem=5760MB
%chk=meoh_974.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4185 0.0187 0.0366
C 0.0258 -0.0210 -0.0030
H 1.7331 0.7398 -0.5478
H -0.3410 0.9423 0.3514
H -0.3825 -0.1336 -1.0074
H -0.3734 -0.8101 0.6342

