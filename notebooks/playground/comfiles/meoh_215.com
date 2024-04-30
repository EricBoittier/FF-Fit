%nproc=4
%mem=5760MB
%chk=meoh_215.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4297 0.1148 0.0028
C -0.0072 -0.0156 0.0019
H 1.7828 -0.7963 -0.0734
H -0.2835 -0.7520 -0.7528
H -0.2939 -0.2805 1.0196
H -0.3530 0.9775 -0.2849

