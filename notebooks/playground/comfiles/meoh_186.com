%nproc=4
%mem=5760MB
%chk=meoh_186.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4211 0.0849 0.0513
C 0.0299 -0.0043 -0.0066
H 1.7200 -0.4336 -0.7248
H -0.3271 -0.8814 -0.5463
H -0.4020 -0.0503 0.9932
H -0.4448 0.8531 -0.4837

