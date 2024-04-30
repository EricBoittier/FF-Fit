%nproc=4
%mem=5760MB
%chk=meoh_961.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4528 -0.0067 -0.0291
C -0.0182 -0.0000 0.0045
H 1.5835 0.9153 0.2765
H -0.3306 1.0257 0.2007
H -0.3367 -0.4264 -0.9468
H -0.2993 -0.6237 0.8530

