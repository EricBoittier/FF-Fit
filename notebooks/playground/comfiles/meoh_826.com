%nproc=4
%mem=5760MB
%chk=meoh_826.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4419 0.0441 -0.0651
C -0.0159 -0.0025 -0.0035
H 1.7261 0.1909 0.8613
H -0.4402 -0.2526 -0.9759
H -0.2120 -0.7776 0.7373
H -0.3119 0.9540 0.4271

