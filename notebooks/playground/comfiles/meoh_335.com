%nproc=4
%mem=5760MB
%chk=meoh_335.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 0.0922 0.0433
C 0.0364 0.0098 -0.0087
H 1.6637 -0.6073 -0.6007
H -0.5156 0.5299 -0.7916
H -0.2718 -1.0305 -0.1129
H -0.4904 0.3132 0.8959

