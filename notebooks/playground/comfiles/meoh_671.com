%nproc=4
%mem=5760MB
%chk=meoh_671.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4276 -0.0004 0.0146
C 0.0023 0.0041 0.0177
H 1.7541 0.8134 -0.4232
H -0.2813 -0.9998 -0.2984
H -0.3783 0.2333 1.0130
H -0.3224 0.6965 -0.7590

