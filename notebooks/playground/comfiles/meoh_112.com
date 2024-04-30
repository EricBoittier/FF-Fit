%nproc=4
%mem=5760MB
%chk=meoh_112.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4385 -0.0114 0.0087
C 0.0048 -0.0002 0.0075
H 1.6516 0.9214 -0.2038
H -0.3865 -1.0133 -0.0853
H -0.3951 0.4804 0.9004
H -0.3015 0.5788 -0.8637

