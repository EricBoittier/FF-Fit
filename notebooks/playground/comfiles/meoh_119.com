%nproc=4
%mem=5760MB
%chk=meoh_119.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4163 -0.0003 0.0198
C 0.0266 -0.0038 0.0009
H 1.7973 0.8324 -0.3294
H -0.3970 -1.0010 -0.1186
H -0.3784 0.4130 0.9230
H -0.3615 0.5907 -0.8261

