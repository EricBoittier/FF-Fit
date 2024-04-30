%nproc=4
%mem=5760MB
%chk=meoh_351.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4293 0.1117 -0.0317
C -0.0083 -0.0208 0.0081
H 1.7839 -0.7331 0.3163
H -0.2901 0.8423 -0.5951
H -0.3180 -0.9895 -0.3841
H -0.3043 0.1400 1.0447

