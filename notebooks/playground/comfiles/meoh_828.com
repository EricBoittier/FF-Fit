%nproc=4
%mem=5760MB
%chk=meoh_828.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4445 0.0391 -0.0668
C -0.0191 -0.0004 0.0012
H 1.7033 0.2541 0.8537
H -0.4163 -0.2376 -0.9858
H -0.2151 -0.8002 0.7153
H -0.3144 0.9532 0.4387

