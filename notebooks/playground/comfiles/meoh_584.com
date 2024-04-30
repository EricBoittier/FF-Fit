%nproc=4
%mem=5760MB
%chk=meoh_584.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4339 0.0790 -0.0703
C 0.0010 -0.0177 0.0152
H 1.7209 -0.2173 0.8188
H -0.3103 -0.9974 0.3778
H -0.3235 0.7744 0.6899
H -0.4013 0.1830 -0.9777

