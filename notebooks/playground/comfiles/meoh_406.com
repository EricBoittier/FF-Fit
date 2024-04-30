%nproc=4
%mem=5760MB
%chk=meoh_406.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4289 0.0362 0.0551
C 0.0269 0.0031 0.0036
H 1.6456 0.2808 -0.8689
H -0.4750 0.8912 0.3875
H -0.3249 -0.1580 -1.0154
H -0.3877 -0.8408 0.5548

