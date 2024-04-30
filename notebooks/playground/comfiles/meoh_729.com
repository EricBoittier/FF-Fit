%nproc=4
%mem=5760MB
%chk=meoh_729.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4183 0.0743 0.0471
C 0.0289 -0.0073 -0.0012
H 1.7844 -0.2218 -0.8124
H -0.3450 -0.8577 -0.5714
H -0.3884 -0.1272 0.9986
H -0.4476 0.8996 -0.3736

