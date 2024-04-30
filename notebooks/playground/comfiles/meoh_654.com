%nproc=4
%mem=5760MB
%chk=meoh_654.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4460 -0.0060 -0.0004
C -0.0139 -0.0085 -0.0002
H 1.6727 0.9437 -0.0847
H -0.3892 -1.0222 -0.1403
H -0.2589 0.3833 0.9869
H -0.3515 0.6760 -0.7785

