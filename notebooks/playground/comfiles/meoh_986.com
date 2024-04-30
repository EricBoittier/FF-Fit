%nproc=4
%mem=5760MB
%chk=meoh_986.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4404 0.0465 0.0519
C 0.0025 0.0018 0.0135
H 1.6673 0.1314 -0.8977
H -0.4938 0.8157 0.5420
H -0.2448 0.0302 -1.0477
H -0.3624 -0.9524 0.3936

