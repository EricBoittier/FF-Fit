%nproc=4
%mem=5760MB
%chk=meoh_926.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4150 0.1047 -0.0303
C 0.0357 0.0044 0.0003
H 1.8044 -0.7081 0.3546
H -0.5715 0.7941 -0.4422
H -0.3444 -0.8950 -0.4842
H -0.3154 -0.1207 1.0245

