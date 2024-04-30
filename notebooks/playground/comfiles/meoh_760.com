%nproc=4
%mem=5760MB
%chk=meoh_760.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4377 0.1144 0.0211
C -0.0073 -0.0163 -0.0074
H 1.7451 -0.7323 -0.3650
H -0.3362 -0.7519 -0.7415
H -0.2699 -0.3360 1.0010
H -0.4129 0.9830 -0.1652

