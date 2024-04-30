%nproc=4
%mem=5760MB
%chk=meoh_782.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4317 0.1174 -0.0258
C -0.0052 -0.0159 0.0125
H 1.7634 -0.7808 0.1829
H -0.2758 -0.6070 -0.8625
H -0.3083 -0.5080 0.9366
H -0.3825 1.0062 -0.0212

