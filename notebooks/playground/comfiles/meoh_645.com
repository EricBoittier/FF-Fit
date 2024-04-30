%nproc=4
%mem=5760MB
%chk=meoh_645.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4202 -0.0032 -0.0209
C 0.0238 -0.0030 0.0161
H 1.7655 0.9039 0.1147
H -0.3819 -1.0066 -0.1113
H -0.3871 0.4054 0.9394
H -0.3642 0.5686 -0.8271

