%nproc=4
%mem=5760MB
%chk=meoh_913.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4366 0.1174 0.0140
C -0.0099 -0.0240 0.0102
H 1.7179 -0.7465 -0.3535
H -0.3288 0.7927 -0.6375
H -0.2243 -1.0152 -0.3894
H -0.3909 0.1767 1.0115

