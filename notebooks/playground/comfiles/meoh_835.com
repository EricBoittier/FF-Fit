%nproc=4
%mem=5760MB
%chk=meoh_835.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4244 0.0241 -0.0664
C 0.0231 -0.0010 0.0209
H 1.7169 0.4649 0.7587
H -0.3897 -0.1194 -0.9810
H -0.3334 -0.8517 0.6016
H -0.4184 0.9204 0.4007

