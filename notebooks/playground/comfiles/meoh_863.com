%nproc=4
%mem=5760MB
%chk=meoh_863.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4206 -0.0071 0.0006
C 0.0333 -0.0021 -0.0012
H 1.7171 0.9219 -0.0966
H -0.4640 0.2174 -0.9460
H -0.3454 -0.9605 0.3541
H -0.3942 0.7446 0.6680

