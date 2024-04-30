%nproc=4
%mem=5760MB
%chk=meoh_104.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4394 -0.0131 0.0001
C -0.0007 0.0026 0.0025
H 1.6708 0.9384 -0.0406
H -0.3595 -1.0265 -0.0186
H -0.3625 0.5153 0.8938
H -0.3291 0.5346 -0.8904

