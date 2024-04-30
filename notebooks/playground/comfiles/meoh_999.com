%nproc=4
%mem=5760MB
%chk=meoh_999.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4294 0.1128 0.0314
C -0.0018 -0.0228 -0.0063
H 1.7467 -0.6403 -0.5096
H -0.3233 0.6511 0.7878
H -0.3792 0.3086 -0.9737
H -0.2518 -1.0529 0.2478

