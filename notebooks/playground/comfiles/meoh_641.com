%nproc=4
%mem=5760MB
%chk=meoh_641.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4176 -0.0057 -0.0299
C 0.0304 0.0032 0.0196
H 1.7506 0.8850 0.2075
H -0.3419 -1.0170 -0.0729
H -0.4354 0.4203 0.9125
H -0.3775 0.5479 -0.8318

