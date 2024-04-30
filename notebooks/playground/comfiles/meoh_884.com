%nproc=4
%mem=5760MB
%chk=meoh_884.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4254 0.0254 0.0492
C 0.0381 -0.0073 -0.0004
H 1.6633 0.4545 -0.7992
H -0.4298 0.4629 -0.8653
H -0.4022 -1.0021 0.0679
H -0.4516 0.5539 0.7954

