%nproc=4
%mem=5760MB
%chk=meoh_797.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4190 0.0974 -0.0435
C 0.0253 -0.0038 0.0010
H 1.8204 -0.5837 0.5357
H -0.4256 -0.4866 -0.8660
H -0.2738 -0.5969 0.8652
H -0.4879 0.9507 0.1176

