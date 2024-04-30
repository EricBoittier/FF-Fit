%nproc=4
%mem=5760MB
%chk=meoh_415.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4417 0.0795 0.0564
C -0.0167 -0.0225 -0.0065
H 1.6803 -0.2630 -0.8303
H -0.2953 0.8766 0.5432
H -0.2889 0.1172 -1.0527
H -0.3216 -0.9400 0.4967

