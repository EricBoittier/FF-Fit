%nproc=4
%mem=5760MB
%chk=meoh_861.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4223 -0.0090 -0.0031
C 0.0372 -0.0000 -0.0058
H 1.6973 0.9315 -0.0173
H -0.5064 0.1871 -0.9319
H -0.3545 -0.9388 0.3856
H -0.3962 0.7490 0.6569

