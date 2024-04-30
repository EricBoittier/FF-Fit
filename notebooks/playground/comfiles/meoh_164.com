%nproc=4
%mem=5760MB
%chk=meoh_164.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4253 0.0453 0.0601
C 0.0224 0.0069 -0.0067
H 1.7594 0.0810 -0.8606
H -0.3164 -0.9531 -0.3963
H -0.4205 0.0899 0.9858
H -0.4539 0.7656 -0.6277

