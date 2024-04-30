%nproc=4
%mem=5760MB
%chk=meoh_547.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4258 0.1109 -0.0074
C -0.0080 -0.0085 0.0113
H 1.8310 -0.7802 -0.0565
H -0.2324 -0.8569 0.6579
H -0.3986 0.9369 0.3878
H -0.2774 -0.1730 -1.0320

