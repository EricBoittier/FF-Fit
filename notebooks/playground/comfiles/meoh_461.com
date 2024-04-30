%nproc=4
%mem=5760MB
%chk=meoh_461.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4457 0.0264 -0.0737
C 0.0012 -0.0024 0.0174
H 1.6007 0.4256 0.8079
H -0.3508 0.1682 1.0348
H -0.3963 0.7871 -0.6205
H -0.3566 -0.9866 -0.2847

