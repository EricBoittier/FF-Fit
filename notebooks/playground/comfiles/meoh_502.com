%nproc=4
%mem=5760MB
%chk=meoh_502.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4378 0.0257 0.0474
C -0.0167 0.0061 0.0035
H 1.7278 0.4028 -0.8095
H -0.3045 -0.4569 0.9474
H -0.3744 1.0274 -0.1274
H -0.2116 -0.6692 -0.8296

