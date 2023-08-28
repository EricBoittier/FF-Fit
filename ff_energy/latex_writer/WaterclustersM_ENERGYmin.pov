#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -11.85*x up 9.51*y
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
atom(<  1.18,   2.14,  -6.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.32,   1.80,  -6.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  1.20,   1.71,  -7.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(<  0.69,  -1.16,  -5.23>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(<  0.93,  -0.77,  -4.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -0.12,  -0.60,  -5.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  2.11,   4.27,  -4.94>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  1.82,   3.55,  -5.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  1.69,   3.94,  -4.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -4.59,  -2.32,  -4.41>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -4.34,  -1.32,  -4.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -5.52,  -2.24,  -4.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  0.44,   1.90,  -9.51>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -0.06,   2.73,  -9.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -0.29,   1.29,  -9.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.37,   2.72,  -4.44>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -2.41,   3.68,  -4.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -3.02,   2.38,  -3.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  3.15,   0.90,  -4.14>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.58,   0.82,  -3.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.67,   1.35,  -4.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.91,  -3.37,  -4.33>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -0.58,  -2.51,  -4.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -0.50,  -3.45,  -3.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -1.30,   0.59,  -5.68>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -1.87,   1.30,  -5.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -2.15,   0.12,  -5.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  5.34,   1.85,  -2.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  5.52,   2.39,  -3.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  4.54,   1.40,  -3.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -0.50,   1.48,  -0.66>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -1.34,   1.42,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  0.14,   2.04,  -0.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(< -1.58,  -0.17,  -9.78>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(< -2.46,   0.17, -10.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -1.80,  -1.01,  -9.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(< -4.02,   0.40,  -4.20>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(< -4.46,   1.23,  -4.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(< -4.11,   0.42,  -5.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -3.79,  -0.41,  -7.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -4.63,  -0.23,  -7.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -3.58,  -1.24,  -7.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(< -2.76,  -4.27,  -6.18>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(< -3.30,  -3.49,  -6.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -2.06,  -3.85,  -5.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -0.04,  -3.07,  -8.40>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(<  0.65,  -2.83,  -9.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(<  0.60,  -3.09,  -7.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(<  4.66,   0.20,  -7.48>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(<  5.38,  -0.43,  -7.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  3.87,  -0.37,  -7.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(<  4.18,   2.86,  -7.07>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(<  4.21,   1.89,  -7.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(<  5.04,   2.88,  -6.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  0.42,  -3.27,  -1.56>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  0.74,  -3.46,  -0.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(< -0.46,  -3.70,  -1.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -2.57,   2.12,  -8.31>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -2.35,   1.18,  -8.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -3.18,   2.27,  -7.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  1.18,   2.14,  -6.41>, <  1.19,   1.92,  -6.84>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.20,   1.71,  -7.27>, <  1.19,   1.92,  -6.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.18,   2.14,  -6.41>, <  0.75,   1.97,  -6.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.32,   1.80,  -6.14>, <  0.75,   1.97,  -6.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,  -1.16,  -5.23>, <  0.81,  -0.97,  -4.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.93,  -0.77,  -4.38>, <  0.81,  -0.97,  -4.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,  -1.16,  -5.23>, <  0.29,  -0.88,  -5.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.12,  -0.60,  -5.29>, <  0.29,  -0.88,  -5.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,   4.27,  -4.94>, <  1.96,   3.91,  -5.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.82,   3.55,  -5.58>, <  1.96,   3.91,  -5.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.11,   4.27,  -4.94>, <  1.90,   4.10,  -4.53>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.69,   3.94,  -4.12>, <  1.90,   4.10,  -4.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.59,  -2.32,  -4.41>, < -5.06,  -2.28,  -4.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.52,  -2.24,  -4.83>, < -5.06,  -2.28,  -4.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.59,  -2.32,  -4.41>, < -4.47,  -1.82,  -4.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.34,  -1.32,  -4.50>, < -4.47,  -1.82,  -4.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.44,   1.90,  -9.51>, <  0.07,   1.59,  -9.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.29,   1.29,  -9.49>, <  0.07,   1.59,  -9.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.44,   1.90,  -9.51>, <  0.19,   2.32,  -9.63>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.06,   2.73,  -9.74>, <  0.19,   2.32,  -9.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.37,   2.72,  -4.44>, < -2.69,   2.55,  -4.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.02,   2.38,  -3.82>, < -2.69,   2.55,  -4.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.37,   2.72,  -4.44>, < -2.39,   3.20,  -4.40>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.41,   3.68,  -4.36>, < -2.39,   3.20,  -4.40>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,   0.90,  -4.14>, <  2.91,   1.12,  -4.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.67,   1.35,  -4.82>, <  2.91,   1.12,  -4.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,   0.90,  -4.14>, <  2.87,   0.86,  -3.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,   0.82,  -3.39>, <  2.87,   0.86,  -3.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.91,  -3.37,  -4.33>, < -0.75,  -2.94,  -4.48>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.58,  -2.51,  -4.62>, < -0.75,  -2.94,  -4.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.91,  -3.37,  -4.33>, < -0.71,  -3.41,  -3.89>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.50,  -3.45,  -3.44>, < -0.71,  -3.41,  -3.89>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.30,   0.59,  -5.68>, < -1.73,   0.35,  -5.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.15,   0.12,  -5.92>, < -1.73,   0.35,  -5.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.30,   0.59,  -5.68>, < -1.59,   0.94,  -5.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.87,   1.30,  -5.31>, < -1.59,   0.94,  -5.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.34,   1.85,  -2.86>, <  4.94,   1.62,  -3.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.54,   1.40,  -3.28>, <  4.94,   1.62,  -3.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.34,   1.85,  -2.86>, <  5.43,   2.12,  -3.25>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.52,   2.39,  -3.65>, <  5.43,   2.12,  -3.25>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.50,   1.48,  -0.66>, < -0.18,   1.76,  -0.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.14,   2.04,  -0.08>, < -0.18,   1.76,  -0.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.50,   1.48,  -0.66>, < -0.92,   1.45,  -0.33>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,   1.42,   0.00>, < -0.92,   1.45,  -0.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.58,  -0.17,  -9.78>, < -2.02,   0.00,  -9.90>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.46,   0.17, -10.02>, < -2.02,   0.00,  -9.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.58,  -0.17,  -9.78>, < -1.69,  -0.59,  -9.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.80,  -1.01,  -9.44>, < -1.69,  -0.59,  -9.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,   0.40,  -4.20>, < -4.07,   0.41,  -4.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.11,   0.42,  -5.20>, < -4.07,   0.41,  -4.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.02,   0.40,  -4.20>, < -4.24,   0.82,  -4.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.46,   1.23,  -4.04>, < -4.24,   0.82,  -4.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.79,  -0.41,  -7.00>, < -3.68,  -0.82,  -7.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.58,  -1.24,  -7.56>, < -3.68,  -0.82,  -7.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.79,  -0.41,  -7.00>, < -4.21,  -0.32,  -7.27>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.63,  -0.23,  -7.54>, < -4.21,  -0.32,  -7.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.76,  -4.27,  -6.18>, < -3.03,  -3.88,  -6.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.30,  -3.49,  -6.20>, < -3.03,  -3.88,  -6.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.76,  -4.27,  -6.18>, < -2.41,  -4.06,  -5.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.06,  -3.85,  -5.64>, < -2.41,  -4.06,  -5.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.04,  -3.07,  -8.40>, <  0.28,  -3.08,  -8.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.60,  -3.09,  -7.65>, <  0.28,  -3.08,  -8.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.04,  -3.07,  -8.40>, <  0.30,  -2.95,  -8.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -2.83,  -9.13>, <  0.30,  -2.95,  -8.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.66,   0.20,  -7.48>, <  5.02,  -0.11,  -7.58>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.38,  -0.43,  -7.67>, <  5.02,  -0.11,  -7.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.66,   0.20,  -7.48>, <  4.26,  -0.09,  -7.43>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.87,  -0.37,  -7.38>, <  4.26,  -0.09,  -7.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,   2.86,  -7.07>, <  4.61,   2.87,  -6.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.04,   2.88,  -6.56>, <  4.61,   2.87,  -6.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,   2.86,  -7.07>, <  4.20,   2.37,  -7.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.21,   1.89,  -7.17>, <  4.20,   2.37,  -7.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.42,  -3.27,  -1.56>, <  0.58,  -3.36,  -1.10>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.74,  -3.46,  -0.64>, <  0.58,  -3.36,  -1.10>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.42,  -3.27,  -1.56>, < -0.02,  -3.48,  -1.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.46,  -3.70,  -1.54>, < -0.02,  -3.48,  -1.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   2.12,  -8.31>, < -2.87,   2.20,  -7.92>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,   2.27,  -7.52>, < -2.87,   2.20,  -7.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.57,   2.12,  -8.31>, < -2.46,   1.65,  -8.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.35,   1.18,  -8.03>, < -2.46,   1.65,  -8.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
