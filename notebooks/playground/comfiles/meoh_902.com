%nproc=4
%mem=5760MB
%chk=meoh_902.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4184 0.0900 0.0401
C 0.0309 -0.0105 0.0134
H 1.7476 -0.3884 -0.7494
H -0.3743 0.6003 -0.7934
H -0.3340 -1.0110 -0.2188
H -0.4625 0.2804 0.9408

