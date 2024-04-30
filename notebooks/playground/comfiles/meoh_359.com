%nproc=4
%mem=5760MB
%chk=meoh_359.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4251 0.0842 -0.0578
C 0.0406 0.0042 0.0041
H 1.6741 -0.4658 0.7141
H -0.5647 0.8083 -0.4145
H -0.3893 -0.8789 -0.4687
H -0.3642 -0.0651 1.0137

