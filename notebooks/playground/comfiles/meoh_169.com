%nproc=4
%mem=5760MB
%chk=meoh_169.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4316 0.0577 0.0539
C -0.0048 -0.0025 0.0032
H 1.8068 -0.0374 -0.8464
H -0.2731 -0.9524 -0.4592
H -0.4008 0.0684 1.0162
H -0.3396 0.8194 -0.6297

