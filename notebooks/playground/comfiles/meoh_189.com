%nproc=4
%mem=5760MB
%chk=meoh_189.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4256 0.0856 0.0480
C 0.0122 0.0022 -0.0071
H 1.7566 -0.4882 -0.6743
H -0.2726 -0.8890 -0.5664
H -0.3475 -0.0920 1.0175
H -0.4505 0.8693 -0.4785

