%nproc=4
%mem=5760MB
%chk=meoh_352.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4286 0.1102 -0.0359
C -0.0001 -0.0197 0.0073
H 1.7635 -0.7142 0.3750
H -0.3229 0.8469 -0.5696
H -0.3366 -0.9778 -0.3887
H -0.3196 0.1167 1.0405

