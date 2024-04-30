%nproc=4
%mem=5760MB
%chk=meoh_630.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4417 -0.0006 -0.0421
C -0.0109 0.0012 0.0082
H 1.6897 0.8077 0.4534
H -0.3116 -1.0462 0.0342
H -0.2781 0.5098 0.9345
H -0.3942 0.5096 -0.8765

