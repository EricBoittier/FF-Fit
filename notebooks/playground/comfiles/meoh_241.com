%nproc=4
%mem=5760MB
%chk=meoh_241.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4289 0.0971 -0.0452
C -0.0076 -0.0171 0.0038
H 1.8081 -0.5123 0.6221
H -0.3423 -0.5615 -0.8792
H -0.2755 -0.5051 0.9409
H -0.3209 1.0261 -0.0359

