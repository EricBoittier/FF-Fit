%nproc=4
%mem=5760MB
%chk=meoh_197.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4334 0.0929 0.0335
C -0.0140 0.0049 -0.0025
H 1.8270 -0.6176 -0.5149
H -0.2088 -0.8791 -0.6097
H -0.3197 -0.1707 1.0289
H -0.4250 0.9186 -0.4320

