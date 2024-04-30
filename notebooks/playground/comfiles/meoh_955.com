%nproc=4
%mem=5760MB
%chk=meoh_955.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4509 0.0002 -0.0513
C -0.0011 0.0086 0.0091
H 1.5630 0.7398 0.5818
H -0.4349 1.0059 0.0833
H -0.3474 -0.5284 -0.8740
H -0.3375 -0.5375 0.8904

