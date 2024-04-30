%nproc=4
%mem=5760MB
%chk=meoh_250.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4127 0.0705 -0.0569
C 0.0317 -0.0028 0.0025
H 1.8237 -0.2394 0.7772
H -0.4027 -0.4838 -0.8740
H -0.3380 -0.5609 0.8627
H -0.4247 0.9838 0.0824

