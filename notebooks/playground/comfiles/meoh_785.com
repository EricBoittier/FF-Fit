%nproc=4
%mem=5760MB
%chk=meoh_785.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4363 0.1183 -0.0274
C -0.0110 -0.0197 0.0060
H 1.7482 -0.7647 0.2614
H -0.3325 -0.5821 -0.8707
H -0.2556 -0.5220 0.9419
H -0.3671 1.0105 0.0067

