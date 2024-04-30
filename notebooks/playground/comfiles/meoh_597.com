%nproc=4
%mem=5760MB
%chk=meoh_597.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4200 0.0503 -0.0642
C 0.0246 -0.0042 0.0022
H 1.8092 0.1306 0.8317
H -0.3332 -0.9955 0.2804
H -0.3559 0.6586 0.7794
H -0.4943 0.2437 -0.9237

