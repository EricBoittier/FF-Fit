%nproc=4
%mem=5760MB
%chk=meoh_815.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4168 0.0700 -0.0665
C 0.0305 -0.0112 0.0145
H 1.7849 -0.1546 0.8136
H -0.4323 -0.2859 -0.9334
H -0.3294 -0.7156 0.7643
H -0.4167 0.9626 0.2140

