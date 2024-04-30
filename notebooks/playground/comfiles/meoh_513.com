%nproc=4
%mem=5760MB
%chk=meoh_513.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4235 0.0583 0.0546
C 0.0349 -0.0033 0.0036
H 1.6873 -0.0246 -0.8856
H -0.4230 -0.5406 0.8341
H -0.4631 0.9663 0.0016
H -0.3525 -0.5011 -0.8853

