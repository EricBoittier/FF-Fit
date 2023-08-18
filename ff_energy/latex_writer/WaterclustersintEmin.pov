#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -10.69*x up 10.50*y
  direction 50.00*z
  location <0,0,50.00> look_at <0,0,0>}


light_source {<  2.00,   3.00,  40.00> color White
  area_light <0.70, 0, 0>, <0, 0.70, 0>, 3, 3
  adaptive 1 jitter}
// no fog
#declare simple = finish {phong 0.7}
#declare pale = finish {ambient 0.5 diffuse 0.85 roughness 0.001 specular 0.200 }
#declare intermediate = finish {ambient 0.3 diffuse 0.6 specular 0.1 roughness 0.04}
#declare vmd = finish {ambient 0.0 diffuse 0.65 phong 0.1 phong_size 40.0 specular 0.5 }
#declare jmol = finish {ambient 0.2 diffuse 0.6 specular 1 roughness 0.001 metallic}
#declare ase2 = finish {ambient 0.05 brilliance 3 diffuse 0.6 metallic specular 0.7 roughness 0.04 reflection 0.15}
#declare ase3 = finish {ambient 0.15 brilliance 2 diffuse 0.6 metallic specular 1.0 roughness 0.001 reflection 0.0}
#declare glass = finish {ambient 0.05 diffuse 0.3 specular 1.0 roughness 0.001}
#declare glass2 = finish {ambient 0.01 diffuse 0.3 specular 1.0 reflection 0.25 roughness 0.001}
#declare Rcell = 0.050;
#declare Rbond = 0.100;

#macro atom(LOC, R, COL, TRANS, FIN)
  sphere{LOC, R texture{pigment{color COL transmit TRANS} finish{FIN}}}
#end
#macro constrain(LOC, R, COL, TRANS FIN)
union{torus{R, Rcell rotate 45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     torus{R, Rcell rotate -45*z texture{pigment{color COL transmit TRANS} finish{FIN}}}
     translate LOC}
#end

// no cell vertices
atom(<  0.27,   0.48,  -4.67>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.94,   0.31,  -5.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.38,  -0.11,  -5.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -4.33,  -1.80,  -6.54>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -4.97,  -2.19,  -7.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -4.74,  -0.87,  -6.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  2.01,  -2.38,  -9.33>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  2.09,  -1.69,  -8.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  1.30,  -2.89,  -8.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  2.86,   1.61,  -0.08>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(<  1.90,   1.81,  -0.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(<  3.01,   2.02,  -0.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -3.14,   0.08,   0.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -3.22,   0.39,  -0.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.15,  -0.26,  -0.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.34,   1.87,  -0.87>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  0.03,   0.92,  -0.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -0.46,   2.36,  -0.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  0.99,   3.19,  -6.44>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  1.63,   2.65,  -6.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  1.51,   3.47,  -5.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -2.71,  -1.02,  -3.21>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -3.38,  -1.69,  -3.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -2.10,  -1.29,  -3.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.28,   1.64,  -9.37>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  2.08,   2.05,  -9.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  0.64,   1.80, -10.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -1.62,  -0.08,  -9.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -1.34,   0.40,  -9.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -2.46,  -0.41,  -9.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.62,  -0.74,  -6.21>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -2.04,  -0.53,  -7.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -1.76,  -1.71,  -6.18>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -0.98,  -3.25,  -1.36>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -0.86,  -3.84,  -2.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.93,  -3.30,  -1.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -0.96,   4.27,  -8.03>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -0.68,   4.88,  -8.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -0.09,   4.02,  -7.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  3.70,   0.12,  -9.42>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(<  4.59,  -0.25,  -9.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(<  3.61,   0.91,  -8.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(< -2.73,   4.38,  -2.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(< -2.67,   3.40,  -2.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -3.66,   4.47,  -2.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -0.99,  -4.74,  -3.81>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(< -1.67,  -4.31,  -4.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(< -0.16,  -4.45,  -4.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  4.13,   3.59,  -2.18>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  3.30,   3.29,  -2.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  4.77,   3.55,  -2.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  4.33,  -0.47,  -3.92>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  3.81,  -0.55,  -4.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  4.97,  -1.18,  -3.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  2.72,  -3.76,  -1.38>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  3.48,  -3.87,  -1.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  2.08,  -3.23,  -1.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -1.81,   2.41,  -5.28>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -1.21,   3.17,  -5.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -1.11,   1.78,  -5.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.27,   0.48,  -4.67>, < -0.06,   0.19,  -4.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.38,  -0.11,  -5.10>, < -0.06,   0.19,  -4.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.27,   0.48,  -4.67>, <  0.60,   0.39,  -4.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.94,   0.31,  -5.28>, <  0.60,   0.39,  -4.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.33,  -1.80,  -6.54>, < -4.65,  -1.99,  -6.87>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.97,  -2.19,  -7.20>, < -4.65,  -1.99,  -6.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.33,  -1.80,  -6.54>, < -4.53,  -1.33,  -6.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.74,  -0.87,  -6.43>, < -4.53,  -1.33,  -6.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,  -2.38,  -9.33>, <  1.66,  -2.63,  -9.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.30,  -2.89,  -8.91>, <  1.66,  -2.63,  -9.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,  -2.38,  -9.33>, <  2.05,  -2.03,  -9.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.09,  -1.69,  -8.68>, <  2.05,  -2.03,  -9.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,   1.61,  -0.08>, <  2.38,   1.71,  -0.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.90,   1.81,  -0.07>, <  2.38,   1.71,  -0.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,   1.61,  -0.08>, <  2.94,   1.81,  -0.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.01,   2.02,  -0.97>, <  2.94,   1.81,  -0.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.14,   0.08,   0.00>, < -2.65,  -0.09,  -0.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.15,  -0.26,  -0.11>, < -2.65,  -0.09,  -0.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.14,   0.08,   0.00>, < -3.18,   0.23,  -0.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.22,   0.39,  -0.93>, < -3.18,   0.23,  -0.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.34,   1.87,  -0.87>, <  0.19,   1.40,  -0.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,   0.92,  -0.72>, <  0.19,   1.40,  -0.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.34,   1.87,  -0.87>, < -0.06,   2.12,  -0.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.46,   2.36,  -0.41>, < -0.06,   2.12,  -0.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.99,   3.19,  -6.44>, <  1.31,   2.92,  -6.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.63,   2.65,  -6.86>, <  1.31,   2.92,  -6.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.99,   3.19,  -6.44>, <  1.25,   3.33,  -6.04>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,   3.47,  -5.64>, <  1.25,   3.33,  -6.04>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.71,  -1.02,  -3.21>, < -3.04,  -1.36,  -3.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,  -1.69,  -3.38>, < -3.04,  -1.36,  -3.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.71,  -1.02,  -3.21>, < -2.41,  -1.16,  -3.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.10,  -1.29,  -3.89>, < -2.41,  -1.16,  -3.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,   1.64,  -9.37>, <  0.96,   1.72,  -9.75>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.64,   1.80, -10.13>, <  0.96,   1.72,  -9.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,   1.64,  -9.37>, <  1.68,   1.85,  -9.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.08,   2.05,  -9.75>, <  1.68,   1.85,  -9.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,  -0.08,  -9.12>, < -2.04,  -0.24,  -9.27>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.46,  -0.41,  -9.43>, < -2.04,  -0.24,  -9.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,  -0.08,  -9.12>, < -1.48,   0.16,  -9.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,   0.40,  -9.89>, < -1.48,   0.16,  -9.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,  -0.74,  -6.21>, < -1.83,  -0.63,  -6.65>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.04,  -0.53,  -7.08>, < -1.83,  -0.63,  -6.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,  -0.74,  -6.21>, < -1.69,  -1.23,  -6.20>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.76,  -1.71,  -6.18>, < -1.69,  -1.23,  -6.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.98,  -3.25,  -1.36>, < -0.92,  -3.55,  -1.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.86,  -3.84,  -2.10>, < -0.92,  -3.55,  -1.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.98,  -3.25,  -1.36>, < -1.45,  -3.27,  -1.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.93,  -3.30,  -1.13>, < -1.45,  -3.27,  -1.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.96,   4.27,  -8.03>, < -0.82,   4.57,  -8.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.68,   4.88,  -8.74>, < -0.82,   4.57,  -8.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.96,   4.27,  -8.03>, < -0.53,   4.14,  -7.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.09,   4.02,  -7.63>, < -0.53,   4.14,  -7.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,   0.12,  -9.42>, <  3.65,   0.51,  -9.10>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.61,   0.91,  -8.77>, <  3.65,   0.51,  -9.10>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,   0.12,  -9.42>, <  4.14,  -0.06,  -9.25>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.59,  -0.25,  -9.08>, <  4.14,  -0.06,  -9.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.73,   4.38,  -2.52>, < -2.70,   3.89,  -2.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.67,   3.40,  -2.42>, < -2.70,   3.89,  -2.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.73,   4.38,  -2.52>, < -3.19,   4.42,  -2.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.66,   4.47,  -2.72>, < -3.19,   4.42,  -2.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.99,  -4.74,  -3.81>, < -1.33,  -4.52,  -4.09>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.67,  -4.31,  -4.37>, < -1.33,  -4.52,  -4.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.99,  -4.74,  -3.81>, < -0.57,  -4.59,  -4.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.16,  -4.45,  -4.24>, < -0.57,  -4.59,  -4.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.13,   3.59,  -2.18>, <  4.45,   3.57,  -2.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.77,   3.55,  -2.93>, <  4.45,   3.57,  -2.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.13,   3.59,  -2.18>, <  3.71,   3.44,  -2.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.30,   3.29,  -2.61>, <  3.71,   3.44,  -2.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.33,  -0.47,  -3.92>, <  4.07,  -0.51,  -4.32>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.81,  -0.55,  -4.71>, <  4.07,  -0.51,  -4.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.33,  -0.47,  -3.92>, <  4.65,  -0.83,  -3.93>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.97,  -1.18,  -3.94>, <  4.65,  -0.83,  -3.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.72,  -3.76,  -1.38>, <  2.40,  -3.49,  -1.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.08,  -3.23,  -1.85>, <  2.40,  -3.49,  -1.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.72,  -3.76,  -1.38>, <  3.10,  -3.81,  -1.68>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.48,  -3.87,  -1.99>, <  3.10,  -3.81,  -1.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,   2.41,  -5.28>, < -1.46,   2.09,  -5.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.11,   1.78,  -5.08>, < -1.46,   2.09,  -5.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.81,   2.41,  -5.28>, < -1.51,   2.79,  -5.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.21,   3.17,  -5.59>, < -1.51,   2.79,  -5.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
