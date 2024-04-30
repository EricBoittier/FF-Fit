%nproc=4
%mem=5760MB
%chk=meoh_385.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4374 -0.0065 -0.0266
C 0.0195 -0.0166 0.0052
H 1.5714 0.9202 0.2628
H -0.2688 1.0345 0.0201
H -0.4893 -0.4294 -0.8659
H -0.4012 -0.4389 0.9178

