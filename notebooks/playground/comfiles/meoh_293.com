%nproc=4
%mem=5760MB
%chk=meoh_293.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4140 -0.0028 -0.0062
C 0.0298 0.0017 0.0030
H 1.8460 0.8764 0.0212
H -0.4082 -0.0167 -0.9950
H -0.3731 -0.8957 0.4724
H -0.4044 0.8461 0.5383

