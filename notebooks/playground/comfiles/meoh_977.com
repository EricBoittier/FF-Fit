%nproc=4
%mem=5760MB
%chk=meoh_977.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4414 0.0210 0.0486
C 0.0008 -0.0108 -0.0058
H 1.5995 0.6262 -0.7059
H -0.3643 0.9146 0.4397
H -0.2971 -0.1146 -1.0492
H -0.3673 -0.8468 0.5889

