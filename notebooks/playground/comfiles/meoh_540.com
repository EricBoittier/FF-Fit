%nproc=4
%mem=5760MB
%chk=meoh_540.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4342 0.1098 0.0095
C -0.0015 -0.0013 0.0109
H 1.7161 -0.7799 -0.2895
H -0.2206 -0.8251 0.6901
H -0.5038 0.9201 0.3056
H -0.2798 -0.2581 -1.0112

