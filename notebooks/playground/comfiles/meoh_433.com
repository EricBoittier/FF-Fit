%nproc=4
%mem=5760MB
%chk=meoh_433.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4522 0.1166 -0.0247
C -0.0176 -0.0041 0.0156
H 1.5946 -0.8395 0.1368
H -0.4501 0.5556 0.8449
H -0.3772 0.3444 -0.9525
H -0.1497 -1.0775 0.1513

