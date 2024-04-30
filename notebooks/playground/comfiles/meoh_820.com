%nproc=4
%mem=5760MB
%chk=meoh_820.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4208 0.0586 -0.0618
C 0.0234 -0.0092 -0.0029
H 1.7938 0.0010 0.8427
H -0.4986 -0.2597 -0.9264
H -0.2823 -0.7271 0.7582
H -0.3840 0.9502 0.3161

