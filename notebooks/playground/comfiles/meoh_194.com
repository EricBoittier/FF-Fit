%nproc=4
%mem=5760MB
%chk=meoh_194.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4335 0.0886 0.0394
C -0.0157 0.0072 -0.0048
H 1.8173 -0.5698 -0.5769
H -0.1998 -0.8920 -0.5927
H -0.2954 -0.1521 1.0366
H -0.4294 0.9072 -0.4597

