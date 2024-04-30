%nproc=4
%mem=5760MB
%chk=meoh_117.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4196 -0.0028 0.0162
C 0.0258 -0.0041 0.0039
H 1.7660 0.8599 -0.2941
H -0.4073 -0.9977 -0.1116
H -0.3928 0.4319 0.9110
H -0.3478 0.5842 -0.8342

