%nproc=4
%mem=5760MB
%chk=meoh_691.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4295 0.0225 0.0435
C 0.0104 -0.0099 -0.0039
H 1.7394 0.5509 -0.7215
H -0.3914 -0.9555 -0.3679
H -0.3293 0.1343 1.0217
H -0.3736 0.8171 -0.6012

