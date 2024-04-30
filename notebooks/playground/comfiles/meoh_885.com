%nproc=4
%mem=5760MB
%chk=meoh_885.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4264 0.0277 0.0512
C 0.0340 -0.0067 -0.0030
H 1.6734 0.4134 -0.8152
H -0.4390 0.4793 -0.8563
H -0.3802 -1.0129 0.0604
H -0.4406 0.5450 0.8086

