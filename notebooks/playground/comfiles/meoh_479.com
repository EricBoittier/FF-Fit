%nproc=4
%mem=5760MB
%chk=meoh_479.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4468 -0.0139 -0.0056
C -0.0199 0.0041 0.0014
H 1.6416 0.9464 0.0137
H -0.2958 -0.0952 1.0512
H -0.3070 0.9669 -0.4212
H -0.3071 -0.8611 -0.5961

