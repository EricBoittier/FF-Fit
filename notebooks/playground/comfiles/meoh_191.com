%nproc=4
%mem=5760MB
%chk=meoh_191.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4294 0.0863 0.0449
C -0.0017 0.0054 -0.0066
H 1.7844 -0.5218 -0.6368
H -0.2346 -0.8933 -0.5778
H -0.3154 -0.1191 1.0299
H -0.4431 0.8854 -0.4744

