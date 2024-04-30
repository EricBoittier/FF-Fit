%nproc=4
%mem=5760MB
%chk=meoh_474.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4246 -0.0048 -0.0206
C -0.0023 0.0049 0.0050
H 1.8095 0.8557 0.2476
H -0.2689 -0.0501 1.0605
H -0.3784 0.8856 -0.5156
H -0.2870 -0.8885 -0.5507

