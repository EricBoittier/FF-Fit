%nproc=4
%mem=5760MB
%chk=meoh_113.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4345 -0.0098 0.0101
C 0.0099 -0.0012 0.0072
H 1.6730 0.9119 -0.2227
H -0.3941 -1.0087 -0.0916
H -0.3981 0.4713 0.9008
H -0.3099 0.5799 -0.8577

