%nproc=4
%mem=5760MB
%chk=meoh_614.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4319 0.0211 -0.0647
C 0.0041 -0.0079 0.0165
H 1.7201 0.5357 0.7180
H -0.3263 -1.0372 0.1561
H -0.3423 0.6152 0.8410
H -0.3694 0.4303 -0.9090

