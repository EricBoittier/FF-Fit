%nproc=4
%mem=5760MB
%chk=meoh_557.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4205 0.1165 -0.0284
C 0.0298 -0.0207 0.0047
H 1.7807 -0.7446 0.2705
H -0.3755 -0.8577 0.5734
H -0.3870 0.8610 0.4915
H -0.4611 -0.0758 -0.9669

