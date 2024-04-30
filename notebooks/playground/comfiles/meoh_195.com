%nproc=4
%mem=5760MB
%chk=meoh_195.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4340 0.0898 0.0374
C -0.0170 0.0069 -0.0041
H 1.8236 -0.5856 -0.5563
H -0.1975 -0.8891 -0.5980
H -0.2987 -0.1599 1.0356
H -0.4265 0.9124 -0.4518

