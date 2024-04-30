%nproc=4
%mem=5760MB
%chk=meoh_943.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4152 0.0662 -0.0583
C 0.0078 -0.0197 0.0082
H 1.8670 0.0445 0.8111
H -0.3309 0.9875 -0.2348
H -0.3638 -0.6840 -0.7719
H -0.2688 -0.3799 0.9990

