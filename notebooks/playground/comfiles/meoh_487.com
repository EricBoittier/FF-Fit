%nproc=4
%mem=5760MB
%chk=meoh_487.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4303 -0.0044 0.0161
C 0.0329 -0.0097 0.0022
H 1.6221 0.8879 -0.3412
H -0.4470 -0.1677 0.9681
H -0.3430 0.9731 -0.2819
H -0.4688 -0.7230 -0.6517

