%nproc=4
%mem=5760MB
%chk=meoh_653.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4452 -0.0055 -0.0023
C -0.0144 -0.0086 0.0015
H 1.6824 0.9435 -0.0629
H -0.3885 -1.0225 -0.1408
H -0.2578 0.3887 0.9868
H -0.3453 0.6650 -0.7890

