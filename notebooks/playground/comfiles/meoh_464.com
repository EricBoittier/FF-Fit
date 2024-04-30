%nproc=4
%mem=5760MB
%chk=meoh_464.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4271 0.0170 -0.0626
C 0.0261 0.0025 0.0138
H 1.6962 0.5541 0.7117
H -0.3799 0.1010 1.0205
H -0.4479 0.7785 -0.5872
H -0.3723 -0.9484 -0.3400

