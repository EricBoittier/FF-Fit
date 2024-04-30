%nproc=4
%mem=5760MB
%chk=meoh_232.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4347 0.1071 -0.0355
C 0.0151 -0.0050 0.0052
H 1.6885 -0.7170 0.4302
H -0.3439 -0.6467 -0.7995
H -0.3301 -0.4381 0.9441
H -0.5068 0.9461 -0.0993

