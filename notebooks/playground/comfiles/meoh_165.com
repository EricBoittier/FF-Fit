%nproc=4
%mem=5760MB
%chk=meoh_165.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4260 0.0475 0.0587
C 0.0179 0.0055 -0.0044
H 1.7728 0.0576 -0.8579
H -0.3086 -0.9524 -0.4092
H -0.4180 0.0844 0.9915
H -0.4360 0.7746 -0.6293

