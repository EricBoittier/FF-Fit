%nproc=4
%mem=5760MB
%chk=meoh_870.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4338 0.0029 0.0119
C -0.0155 -0.0069 0.0164
H 1.7369 0.8505 -0.3755
H -0.2140 0.2757 -1.0175
H -0.3160 -1.0336 0.2253
H -0.3220 0.7297 0.7589

