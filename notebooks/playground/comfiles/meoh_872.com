%nproc=4
%mem=5760MB
%chk=meoh_872.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4368 0.0058 0.0161
C -0.0202 -0.0072 0.0185
H 1.7226 0.8176 -0.4528
H -0.1773 0.2874 -1.0191
H -0.3308 -1.0378 0.1900
H -0.3208 0.7117 0.7807

