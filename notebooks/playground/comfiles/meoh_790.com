%nproc=4
%mem=5760MB
%chk=meoh_790.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4290 0.1121 -0.0302
C 0.0080 -0.0169 -0.0032
H 1.7748 -0.7065 0.3830
H -0.4438 -0.5238 -0.8559
H -0.2268 -0.5457 0.9206
H -0.4226 0.9833 0.0448

