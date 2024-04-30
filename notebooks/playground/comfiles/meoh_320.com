%nproc=4
%mem=5760MB
%chk=meoh_320.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4187 0.0498 0.0450
C 0.0074 -0.0107 0.0146
H 1.8464 0.1328 -0.8329
H -0.2537 0.3962 -0.9623
H -0.3620 -1.0346 0.0714
H -0.3791 0.6270 0.8096

