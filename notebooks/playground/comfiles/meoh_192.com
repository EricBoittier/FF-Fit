%nproc=4
%mem=5760MB
%chk=meoh_192.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4311 0.0869 0.0431
C -0.0078 0.0064 -0.0061
H 1.7972 -0.5380 -0.6172
H -0.2190 -0.8941 -0.5829
H -0.3041 -0.1314 1.0337
H -0.4382 0.8934 -0.4709

