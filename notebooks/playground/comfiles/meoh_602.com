%nproc=4
%mem=5760MB
%chk=meoh_602.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4207 0.0361 -0.0676
C 0.0212 0.0077 0.0131
H 1.7832 0.2607 0.8148
H -0.2868 -1.0197 0.2076
H -0.3923 0.6267 0.8092
H -0.4481 0.2525 -0.9398

