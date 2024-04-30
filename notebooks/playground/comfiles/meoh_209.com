%nproc=4
%mem=5760MB
%chk=meoh_209.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4253 0.1175 0.0130
C 0.0246 -0.0223 0.0039
H 1.7239 -0.7813 -0.2389
H -0.3638 -0.7558 -0.7028
H -0.4204 -0.2064 0.9818
H -0.3966 0.9298 -0.3189

