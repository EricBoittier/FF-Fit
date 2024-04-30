%nproc=4
%mem=5760MB
%chk=meoh_813.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4212 0.0746 -0.0693
C 0.0207 -0.0110 0.0219
H 1.7686 -0.2153 0.8000
H -0.3756 -0.3107 -0.9484
H -0.3292 -0.7173 0.7747
H -0.4098 0.9763 0.1889

