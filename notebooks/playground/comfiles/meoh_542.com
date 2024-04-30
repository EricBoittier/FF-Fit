%nproc=4
%mem=5760MB
%chk=meoh_542.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4353 0.1104 0.0044
C -0.0127 -0.0035 0.0121
H 1.7473 -0.7902 -0.2241
H -0.1951 -0.8412 0.6854
H -0.4686 0.9355 0.3258
H -0.2541 -0.2288 -1.0267

