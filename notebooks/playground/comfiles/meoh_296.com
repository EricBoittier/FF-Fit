%nproc=4
%mem=5760MB
%chk=meoh_296.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4155 -0.0013 0.0067
C 0.0198 0.0021 -0.0025
H 1.8665 0.8591 -0.1230
H -0.3941 0.0198 -1.0106
H -0.3381 -0.9253 0.4448
H -0.3802 0.8273 0.5867

