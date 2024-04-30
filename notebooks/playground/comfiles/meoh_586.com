%nproc=4
%mem=5760MB
%chk=meoh_586.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4403 0.0773 -0.0713
C -0.0099 -0.0224 0.0134
H 1.7081 -0.1671 0.8392
H -0.3043 -1.0054 0.3810
H -0.2892 0.7752 0.7019
H -0.4001 0.2222 -0.9745

