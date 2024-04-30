%nproc=4
%mem=5760MB
%chk=meoh_446.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4009 0.0783 -0.0453
C 0.0397 -0.0040 -0.0037
H 1.9055 -0.3424 0.6820
H -0.3121 0.4029 0.9444
H -0.4742 0.5398 -0.7964
H -0.3689 -1.0106 -0.0923

