%nproc=4
%mem=5760MB
%chk=meoh_311.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4311 0.0143 0.0494
C 0.0267 -0.0061 0.0009
H 1.6047 0.5705 -0.7386
H -0.3284 0.2995 -0.9833
H -0.4362 -0.9668 0.2264
H -0.4147 0.7277 0.6753

