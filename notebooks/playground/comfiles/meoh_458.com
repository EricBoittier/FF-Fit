%nproc=4
%mem=5760MB
%chk=meoh_458.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4557 0.0380 -0.0781
C -0.0189 -0.0085 0.0176
H 1.5758 0.2800 0.8640
H -0.3001 0.2393 1.0412
H -0.3529 0.7792 -0.6577
H -0.3442 -1.0155 -0.2436

