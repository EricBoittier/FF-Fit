%nproc=4
%mem=5760MB
%chk=meoh_787.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4353 0.1169 -0.0283
C -0.0069 -0.0200 0.0018
H 1.7518 -0.7461 0.3117
H -0.3799 -0.5597 -0.8687
H -0.2335 -0.5310 0.9375
H -0.3795 1.0041 0.0220

