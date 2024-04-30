%nproc=4
%mem=5760MB
%chk=meoh_964.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4316 0.0001 -0.0131
C 0.0022 -0.0124 0.0025
H 1.7117 0.9333 0.0928
H -0.3035 1.0126 0.2122
H -0.3887 -0.3399 -0.9609
H -0.3100 -0.6746 0.8100

