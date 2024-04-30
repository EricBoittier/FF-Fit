%nproc=4
%mem=5760MB
%chk=meoh_277.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4296 0.0091 -0.0467
C 0.0094 -0.0073 0.0037
H 1.7001 0.6800 0.6145
H -0.3894 -0.1572 -0.9996
H -0.3541 -0.7681 0.6945
H -0.3017 0.9735 0.3633

