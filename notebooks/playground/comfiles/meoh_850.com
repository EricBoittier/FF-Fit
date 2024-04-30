%nproc=4
%mem=5760MB
%chk=meoh_850.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4448 -0.0003 -0.0355
C -0.0219 -0.0026 0.0071
H 1.7104 0.8473 0.3786
H -0.3363 0.0294 -1.0361
H -0.3214 -0.9272 0.5005
H -0.2651 0.8715 0.6111

