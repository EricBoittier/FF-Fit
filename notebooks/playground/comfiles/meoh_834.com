%nproc=4
%mem=5760MB
%chk=meoh_834.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4287 0.0258 -0.0673
C 0.0153 -0.0001 0.0186
H 1.7063 0.4363 0.7783
H -0.3906 -0.1394 -0.9834
H -0.3127 -0.8489 0.6186
H -0.4030 0.9282 0.4077

