%nproc=4
%mem=5760MB
%chk=meoh_200.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4297 0.0993 0.0279
C 0.0018 -0.0011 0.0001
H 1.8094 -0.6664 -0.4518
H -0.2586 -0.8523 -0.6289
H -0.3754 -0.1769 1.0075
H -0.4314 0.9173 -0.3961

