%nproc=4
%mem=5760MB
%chk=meoh_912.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4352 0.1173 0.0169
C -0.0016 -0.0259 0.0112
H 1.6942 -0.7305 -0.4010
H -0.3328 0.7818 -0.6415
H -0.2426 -1.0158 -0.3764
H -0.4208 0.1959 0.9926

