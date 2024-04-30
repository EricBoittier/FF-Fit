%nproc=4
%mem=5760MB
%chk=meoh_973.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4120 0.0184 0.0319
C 0.0325 -0.0238 -0.0020
H 1.7739 0.7675 -0.4861
H -0.3346 0.9496 0.3232
H -0.4064 -0.1421 -0.9927
H -0.3735 -0.7980 0.6490

