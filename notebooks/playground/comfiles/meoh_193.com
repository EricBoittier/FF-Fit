%nproc=4
%mem=5760MB
%chk=meoh_193.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4325 0.0876 0.0413
C -0.0125 0.0070 -0.0055
H 1.8084 -0.5539 -0.5972
H -0.2071 -0.8937 -0.5878
H -0.2973 -0.1425 1.0360
H -0.4335 0.9008 -0.4660

