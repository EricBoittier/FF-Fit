%nproc=4
%mem=5760MB
%chk=meoh_565.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4427 0.1071 -0.0455
C -0.0170 -0.0140 0.0097
H 1.7362 -0.6591 0.4906
H -0.2015 -0.9468 0.5427
H -0.3868 0.8491 0.5632
H -0.3866 0.0085 -1.0154

