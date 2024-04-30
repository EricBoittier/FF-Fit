%nproc=4
%mem=5760MB
%chk=meoh_942.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4199 0.0717 -0.0604
C 0.0002 -0.0217 0.0090
H 1.8408 -0.0200 0.8199
H -0.3175 0.9907 -0.2403
H -0.3449 -0.7073 -0.7649
H -0.2603 -0.3571 1.0128

