%nproc=4
%mem=5760MB
%chk=meoh_696.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4327 0.0276 0.0444
C -0.0104 -0.0051 0.0036
H 1.7655 0.4587 -0.7704
H -0.2985 -0.9631 -0.4295
H -0.2953 0.0856 1.0518
H -0.3289 0.8270 -0.6243

