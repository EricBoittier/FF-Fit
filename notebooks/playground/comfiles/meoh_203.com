%nproc=4
%mem=5760MB
%chk=meoh_203.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4253 0.1066 0.0227
C 0.0210 -0.0094 0.0024
H 1.7735 -0.7133 -0.3859
H -0.3223 -0.8140 -0.6478
H -0.4296 -0.1793 0.9803
H -0.4371 0.9104 -0.3613

