%nproc=4
%mem=5760MB
%chk=meoh_337.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4201 0.0964 0.0353
C 0.0399 0.0082 -0.0039
H 1.7062 -0.6671 -0.5084
H -0.5008 0.5529 -0.7779
H -0.2859 -1.0193 -0.1654
H -0.4773 0.2903 0.9132

