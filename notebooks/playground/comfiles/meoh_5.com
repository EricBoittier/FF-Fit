%nproc=4
%mem=5760MB
%chk=meoh_5.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4303 -0.0083 -0.0002
C 0.0037 0.0010 0.0001
H 1.7278 0.9255 0.0001
H -0.3657 -1.0245 -0.0001
H -0.3597 0.5152 0.8899
H -0.3596 0.5149 -0.8898

