%nproc=4
%mem=5760MB
%chk=meoh_289.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4295 -0.0087 -0.0232
C 0.0181 0.0034 0.0103
H 1.7135 0.9003 0.2082
H -0.3842 -0.0439 -1.0017
H -0.3796 -0.8679 0.5307
H -0.3961 0.8941 0.4828

