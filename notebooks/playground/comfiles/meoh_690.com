%nproc=4
%mem=5760MB
%chk=meoh_690.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4288 0.0211 0.0430
C 0.0157 -0.0104 -0.0048
H 1.7304 0.5689 -0.7116
H -0.4071 -0.9518 -0.3556
H -0.3454 0.1435 1.0121
H -0.3849 0.8131 -0.5959

