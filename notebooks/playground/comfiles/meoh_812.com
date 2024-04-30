%nproc=4
%mem=5760MB
%chk=meoh_812.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4240 0.0767 -0.0705
C 0.0143 -0.0108 0.0248
H 1.7604 -0.2448 0.7920
H -0.3453 -0.3256 -0.9548
H -0.3260 -0.7180 0.7812
H -0.4044 0.9835 0.1803

