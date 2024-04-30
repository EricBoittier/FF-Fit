%nproc=4
%mem=5760MB
%chk=meoh_426.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4122 0.1026 0.0046
C 0.0326 0.0043 0.0128
H 1.8502 -0.7290 -0.2732
H -0.4588 0.6492 0.7413
H -0.4200 0.1873 -0.9617
H -0.3157 -1.0027 0.2424

