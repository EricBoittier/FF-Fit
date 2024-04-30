%nproc=4
%mem=5760MB
%chk=meoh_210.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 0.1181 0.0114
C 0.0197 -0.0228 0.0038
H 1.7261 -0.7877 -0.2125
H -0.3548 -0.7524 -0.7144
H -0.4020 -0.2157 0.9902
H -0.3846 0.9381 -0.3144

