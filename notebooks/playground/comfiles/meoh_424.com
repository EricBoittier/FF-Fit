%nproc=4
%mem=5760MB
%chk=meoh_424.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4054 0.0971 0.0134
C 0.0380 0.0019 0.0105
H 1.8982 -0.6508 -0.3847
H -0.4342 0.6910 0.7108
H -0.4162 0.1616 -0.9673
H -0.3497 -0.9809 0.2783

