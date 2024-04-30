%nproc=4
%mem=5760MB
%chk=meoh_150.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4323 0.0306 0.0533
C -0.0079 -0.0076 -0.0020
H 1.7512 0.3909 -0.8005
H -0.3062 -1.0039 -0.3285
H -0.3585 0.2337 1.0014
H -0.2682 0.7701 -0.7199

