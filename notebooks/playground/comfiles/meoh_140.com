%nproc=4
%mem=5760MB
%chk=meoh_140.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4247 0.0156 0.0431
C 0.0256 -0.0047 0.0073
H 1.7363 0.5842 -0.6918
H -0.3939 -0.9682 -0.2823
H -0.4656 0.2940 0.9334
H -0.3358 0.6848 -0.7556

