%nproc=4
%mem=5760MB
%chk=meoh_340.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4166 0.1020 0.0211
C 0.0265 0.0021 0.0040
H 1.7820 -0.7286 -0.3491
H -0.4254 0.6143 -0.7765
H -0.2807 -1.0139 -0.2439
H -0.4181 0.2691 0.9627

