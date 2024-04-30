%nproc=4
%mem=5760MB
%chk=meoh_631.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4420 -0.0026 -0.0418
C -0.0116 0.0030 0.0099
H 1.6829 0.8197 0.4339
H -0.2955 -1.0493 0.0255
H -0.2929 0.5035 0.9365
H -0.3856 0.5166 -0.8758

