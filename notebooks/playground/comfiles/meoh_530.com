%nproc=4
%mem=5760MB
%chk=meoh_530.chk
#P PBE1PBE/aug-cc-pVDZ scf(maxcycle=200) 

Gaussian input

0 1
O 1.4223 0.1016 0.0303
C 0.0254 -0.0070 0.0024
H 1.7597 -0.5922 -0.5742
H -0.3567 -0.7094 0.7431
H -0.4563 0.9417 0.2390
H -0.3673 -0.3843 -0.9418

