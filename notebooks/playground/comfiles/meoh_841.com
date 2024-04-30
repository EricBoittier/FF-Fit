%nproc=4
%mem=5760MB
%chk=meoh_841.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4137 0.0160 -0.0571
C 0.0342 -0.0083 0.0259
H 1.7794 0.6265 0.6167
H -0.3511 -0.0235 -0.9936
H -0.3946 -0.8743 0.5301
H -0.4206 0.9009 0.4193

