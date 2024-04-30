%nproc=4
%mem=5760MB
%chk=meoh_748.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4188 0.0957 0.0280
C 0.0189 -0.0076 0.0131
H 1.8178 -0.5526 -0.5893
H -0.2954 -0.7575 -0.7130
H -0.3857 -0.2657 0.9917
H -0.4229 0.9331 -0.3153

