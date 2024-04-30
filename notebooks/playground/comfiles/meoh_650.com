%nproc=4
%mem=5760MB
%chk=meoh_650.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4378 -0.0039 -0.0089
C -0.0057 -0.0079 0.0073
H 1.7203 0.9344 0.0031
H -0.3898 -1.0176 -0.1386
H -0.2849 0.3990 0.9791
H -0.3405 0.6261 -0.8137

