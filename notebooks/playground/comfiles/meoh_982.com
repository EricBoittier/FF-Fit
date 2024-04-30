%nproc=4
%mem=5760MB
%chk=meoh_982.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4592 0.0323 0.0566
C -0.0190 0.0017 -0.0010
H 1.5236 0.3651 -0.8630
H -0.4311 0.8568 0.5349
H -0.2072 -0.0609 -1.0728
H -0.3609 -0.9097 0.4894

