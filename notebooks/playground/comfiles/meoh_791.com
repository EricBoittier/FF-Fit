%nproc=4
%mem=5760MB
%chk=meoh_791.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4263 0.1100 -0.0313
C 0.0137 -0.0151 -0.0042
H 1.7849 -0.6907 0.4057
H -0.4583 -0.5131 -0.8511
H -0.2308 -0.5512 0.9128
H -0.4393 0.9746 0.0533

