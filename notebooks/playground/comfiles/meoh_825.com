%nproc=4
%mem=5760MB
%chk=meoh_825.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4391 0.0466 -0.0641
C -0.0113 -0.0038 -0.0052
H 1.7396 0.1591 0.8619
H -0.4538 -0.2566 -0.9687
H -0.2170 -0.7670 0.7454
H -0.3176 0.9540 0.4156

