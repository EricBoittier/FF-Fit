%nproc=4
%mem=5760MB
%chk=meoh_600.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4178 0.0415 -0.0654
C 0.0274 0.0041 0.0076
H 1.8035 0.2091 0.8198
H -0.3126 -1.0064 0.2344
H -0.3854 0.6329 0.7965
H -0.4775 0.2418 -0.9287

