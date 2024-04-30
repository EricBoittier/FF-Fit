%nproc=4
%mem=5760MB
%chk=meoh_176.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4326 0.0751 0.0522
C -0.0044 -0.0150 0.0036
H 1.7598 -0.2105 -0.8264
H -0.2927 -0.9327 -0.5091
H -0.4176 0.0446 1.0104
H -0.2778 0.8704 -0.5703

