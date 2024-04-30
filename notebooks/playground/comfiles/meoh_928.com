%nproc=4
%mem=5760MB
%chk=meoh_928.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4199 0.1059 -0.0398
C 0.0376 -0.0019 0.0017
H 1.7332 -0.6757 0.4616
H -0.5712 0.8169 -0.3819
H -0.3535 -0.8811 -0.5104
H -0.3346 -0.1341 1.0176

