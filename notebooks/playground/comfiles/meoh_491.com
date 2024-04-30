%nproc=4
%mem=5760MB
%chk=meoh_491.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4199 0.0067 0.0237
C 0.0331 -0.0149 0.0073
H 1.7360 0.7815 -0.4865
H -0.4237 -0.2289 0.9736
H -0.3443 0.9738 -0.2536
H -0.4418 -0.6700 -0.7231

