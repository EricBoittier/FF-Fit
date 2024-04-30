%nproc=4
%mem=5760MB
%chk=meoh_452.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4308 0.0606 -0.0655
C -0.0009 -0.0134 0.0065
H 1.7652 -0.0293 0.8514
H -0.2457 0.3478 1.0054
H -0.3866 0.6831 -0.7379
H -0.3739 -1.0199 -0.1827

