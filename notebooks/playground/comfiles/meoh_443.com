%nproc=4
%mem=5760MB
%chk=meoh_443.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4080 0.0894 -0.0421
C 0.0330 -0.0012 -0.0008
H 1.8500 -0.5096 0.5953
H -0.3560 0.4317 0.9209
H -0.4696 0.4942 -0.8316
H -0.3069 -1.0364 -0.0314

