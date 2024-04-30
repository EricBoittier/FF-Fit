%nproc=4
%mem=5760MB
%chk=meoh_833.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4330 0.0277 -0.0679
C 0.0071 0.0006 0.0160
H 1.6979 0.4071 0.7961
H -0.3914 -0.1592 -0.9859
H -0.2914 -0.8447 0.6360
H -0.3856 0.9355 0.4157

