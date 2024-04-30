%nproc=4
%mem=5760MB
%chk=meoh_216.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4293 0.1131 0.0009
C -0.0100 -0.0125 0.0013
H 1.7985 -0.7936 -0.0446
H -0.2714 -0.7534 -0.7543
H -0.2778 -0.2958 1.0192
H -0.3580 0.9825 -0.2759

