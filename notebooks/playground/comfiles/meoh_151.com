%nproc=4
%mem=5760MB
%chk=meoh_151.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4336 0.0311 0.0554
C -0.0083 -0.0061 -0.0048
H 1.7387 0.3708 -0.8118
H -0.2997 -1.0057 -0.3274
H -0.3530 0.2230 1.0036
H -0.2827 0.7762 -0.7124

