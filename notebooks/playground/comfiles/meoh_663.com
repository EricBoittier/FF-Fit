%nproc=4
%mem=5760MB
%chk=meoh_663.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4234 -0.0050 0.0093
C 0.0295 -0.0017 0.0017
H 1.7095 0.8901 -0.2692
H -0.3902 -0.9913 -0.1793
H -0.3864 0.2942 0.9648
H -0.4175 0.6919 -0.7105

