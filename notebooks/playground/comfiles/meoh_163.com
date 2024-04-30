%nproc=4
%mem=5760MB
%chk=meoh_163.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4250 0.0432 0.0613
C 0.0255 0.0080 -0.0089
H 1.7454 0.1043 -0.8629
H -0.3222 -0.9547 -0.3839
H -0.4211 0.0962 0.9815
H -0.4668 0.7589 -0.6269

