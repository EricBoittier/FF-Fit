%nproc=4
%mem=5760MB
%chk=meoh_853.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4454 -0.0057 -0.0257
C -0.0131 0.0011 -0.0014
H 1.6759 0.8959 0.2817
H -0.4043 0.0589 -1.0172
H -0.3210 -0.9251 0.4839
H -0.2774 0.8332 0.6511

