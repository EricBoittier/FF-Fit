%nproc=4
%mem=5760MB
%chk=meoh_103.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4355 -0.0119 -0.0008
C 0.0038 0.0021 0.0012
H 1.6953 0.9329 -0.0194
H -0.3631 -1.0242 -0.0091
H -0.3601 0.5153 0.8913
H -0.3437 0.5244 -0.8901

