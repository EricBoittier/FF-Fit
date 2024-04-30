%nproc=4
%mem=5760MB
%chk=meoh_400.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4131 0.0110 0.0297
C 0.0314 0.0160 0.0139
H 1.8193 0.6092 -0.6319
H -0.4938 0.9286 0.2959
H -0.3086 -0.3300 -0.9622
H -0.3616 -0.7878 0.6365

