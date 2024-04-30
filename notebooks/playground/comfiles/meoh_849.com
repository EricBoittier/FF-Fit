%nproc=4
%mem=5760MB
%chk=meoh_849.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4425 0.0017 -0.0386
C -0.0203 -0.0041 0.0101
H 1.7246 0.8270 0.4084
H -0.3213 0.0231 -1.0372
H -0.3267 -0.9261 0.5041
H -0.2719 0.8827 0.5917

