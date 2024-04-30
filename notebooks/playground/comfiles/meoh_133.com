%nproc=4
%mem=5760MB
%chk=meoh_133.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4420 0.0008 0.0448
C -0.0004 0.0076 -0.0089
H 1.6590 0.7013 -0.6054
H -0.3181 -1.0178 -0.1982
H -0.3889 0.3242 0.9590
H -0.3771 0.6743 -0.7846

