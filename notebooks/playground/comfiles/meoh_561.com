%nproc=4
%mem=5760MB
%chk=meoh_561.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4358 0.1149 -0.0375
C 0.0038 -0.0211 0.0064
H 1.7315 -0.7164 0.3891
H -0.2950 -0.9052 0.5696
H -0.3782 0.8575 0.5263
H -0.4352 -0.0232 -0.9912

