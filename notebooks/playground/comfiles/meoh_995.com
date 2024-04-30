%nproc=4
%mem=5760MB
%chk=meoh_995.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4100 0.0890 0.0341
C 0.0250 -0.0155 0.0118
H 1.8814 -0.4139 -0.6626
H -0.4285 0.7402 0.6532
H -0.3803 0.2581 -0.9623
H -0.2920 -1.0272 0.2649

