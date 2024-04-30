%nproc=4
%mem=5760MB
%chk=meoh_806.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4354 0.0870 -0.0694
C -0.0122 -0.0059 0.0271
H 1.7502 -0.4047 0.7177
H -0.2352 -0.4271 -0.9533
H -0.2968 -0.6972 0.8203
H -0.3972 1.0039 0.1693

