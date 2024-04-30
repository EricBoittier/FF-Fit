%nproc=4
%mem=5760MB
%chk=meoh_874.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4372 0.0087 0.0210
C -0.0157 -0.0076 0.0190
H 1.7026 0.7770 -0.5265
H -0.1755 0.3057 -1.0127
H -0.3570 -1.0331 0.1600
H -0.3372 0.6879 0.7942

