%nproc=4
%mem=5760MB
%chk=meoh_349.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4294 0.1130 -0.0229
C -0.0198 -0.0206 0.0097
H 1.8208 -0.7586 0.1951
H -0.2468 0.8202 -0.6457
H -0.2851 -1.0054 -0.3750
H -0.2831 0.1807 1.0481

