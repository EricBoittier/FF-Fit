%nproc=4
%mem=5760MB
%chk=meoh_739.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4405 0.0889 0.0409
C -0.0088 -0.0051 0.0113
H 1.6887 -0.4360 -0.7486
H -0.2121 -0.8239 -0.6790
H -0.3783 -0.2360 1.0104
H -0.3985 0.9298 -0.3915

