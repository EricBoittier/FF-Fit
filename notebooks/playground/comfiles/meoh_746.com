%nproc=4
%mem=5760MB
%chk=meoh_746.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4253 0.0943 0.0305
C 0.0072 -0.0068 0.0160
H 1.7908 -0.5307 -0.6301
H -0.2592 -0.7659 -0.7195
H -0.3764 -0.2658 1.0029
H -0.4058 0.9320 -0.3529

