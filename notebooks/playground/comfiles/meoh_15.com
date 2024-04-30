%nproc=4
%mem=5760MB
%chk=meoh_15.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4312 -0.0104 -0.0003
C 0.0087 0.0018 0.0001
H 1.7219 0.9255 0.0003
H -0.3669 -1.0215 -0.0000
H -0.3590 0.5144 0.8890
H -0.3591 0.5141 -0.8890

