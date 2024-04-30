%nproc=4
%mem=5760MB
%chk=meoh_775.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4140 0.1096 -0.0160
C 0.0288 -0.0017 0.0223
H 1.8212 -0.7818 -0.0047
H -0.2567 -0.6270 -0.8236
H -0.3983 -0.4713 0.9083
H -0.4935 0.9450 -0.1161

