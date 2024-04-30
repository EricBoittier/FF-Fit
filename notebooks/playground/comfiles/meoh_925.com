%nproc=4
%mem=5760MB
%chk=meoh_925.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4142 0.1042 -0.0257
C 0.0312 0.0064 0.0001
H 1.8352 -0.7191 0.2989
H -0.5607 0.7896 -0.4737
H -0.3323 -0.9049 -0.4746
H -0.3020 -0.1114 1.0312

