%nproc=4
%mem=5760MB
%chk=meoh_893.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4379 0.0529 0.0543
C -0.0255 -0.0008 -0.0059
H 1.7990 0.0488 -0.8568
H -0.3633 0.5557 -0.8801
H -0.2038 -1.0756 -0.0408
H -0.2932 0.4262 0.9606

