%nproc=4
%mem=5760MB
%chk=meoh_498.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4317 0.0201 0.0376
C -0.0166 -0.0049 0.0091
H 1.7995 0.5458 -0.7033
H -0.2958 -0.3795 0.9939
H -0.3226 1.0215 -0.1933
H -0.2492 -0.6636 -0.8277

