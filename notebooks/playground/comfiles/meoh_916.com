%nproc=4
%mem=5760MB
%chk=meoh_916.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4355 0.1144 0.0053
C -0.0234 -0.0139 0.0074
H 1.8041 -0.7700 -0.2007
H -0.3459 0.8088 -0.6308
H -0.1978 -1.0015 -0.4196
H -0.3082 0.0980 1.0535

