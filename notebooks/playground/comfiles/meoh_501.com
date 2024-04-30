%nproc=4
%mem=5760MB
%chk=meoh_501.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4372 0.0241 0.0451
C -0.0202 0.0036 0.0050
H 1.7491 0.4397 -0.7859
H -0.2944 -0.4402 0.9620
H -0.3565 1.0295 -0.1449
H -0.2116 -0.6696 -0.8306

