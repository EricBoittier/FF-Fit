%nproc=4
%mem=5760MB
%chk=meoh_655.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4457 -0.0064 0.0014
C -0.0116 -0.0082 -0.0017
H 1.6656 0.9426 -0.1063
H -0.3907 -1.0208 -0.1399
H -0.2649 0.3768 0.9861
H -0.3596 0.6851 -0.7674

