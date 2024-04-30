%nproc=4
%mem=5760MB
%chk=meoh_124.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4249 0.0002 0.0309
C 0.0073 0.0021 -0.0081
H 1.7877 0.7859 -0.4292
H -0.3319 -1.0265 -0.1313
H -0.3304 0.3714 0.9602
H -0.3713 0.6259 -0.8178

