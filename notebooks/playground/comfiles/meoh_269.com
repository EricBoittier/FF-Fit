%nproc=4
%mem=5760MB
%chk=meoh_269.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4121 0.0316 -0.0493
C 0.0261 -0.0049 -0.0060
H 1.8547 0.3869 0.7497
H -0.4435 -0.2964 -0.9455
H -0.3196 -0.7219 0.7387
H -0.3569 0.9732 0.2850

