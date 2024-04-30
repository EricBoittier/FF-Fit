%nproc=4
%mem=5760MB
%chk=meoh_202.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4266 0.1042 0.0244
C 0.0151 -0.0065 0.0018
H 1.7864 -0.6982 -0.4083
H -0.3018 -0.8274 -0.6414
H -0.4140 -0.1783 0.9889
H -0.4364 0.9124 -0.3722

