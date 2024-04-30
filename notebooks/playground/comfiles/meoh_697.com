%nproc=4
%mem=5760MB
%chk=meoh_697.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4326 0.0284 0.0444
C -0.0110 -0.0040 0.0053
H 1.7652 0.4400 -0.7805
H -0.2837 -0.9612 -0.4392
H -0.3031 0.0765 1.0523
H -0.3274 0.8273 -0.6247

