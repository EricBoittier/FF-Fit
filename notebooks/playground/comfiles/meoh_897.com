%nproc=4
%mem=5760MB
%chk=meoh_897.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4312 0.0688 0.0487
C -0.0149 -0.0008 0.0032
H 1.8161 -0.1462 -0.8266
H -0.3399 0.5603 -0.8729
H -0.2332 -1.0610 -0.1249
H -0.3239 0.3499 0.9879

