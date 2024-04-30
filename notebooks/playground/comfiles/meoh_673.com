%nproc=4
%mem=5760MB
%chk=meoh_673.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4327 0.0004 0.0173
C -0.0073 0.0039 0.0183
H 1.7424 0.7971 -0.4622
H -0.2656 -1.0000 -0.3188
H -0.3694 0.2262 1.0221
H -0.3033 0.7084 -0.7589

