#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -12.24*x up 9.53*y
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
atom(<  0.06,  -0.18,  -4.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -0.23,  -0.81,  -3.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(<  0.69,   0.33,  -4.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -2.78,   2.67,  -1.94>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -3.62,   2.98,  -1.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.54,   3.47,  -2.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  0.75,  -0.80,  -7.22>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  0.55,  -0.11,  -7.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  0.43,  -0.50,  -6.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -0.41,   1.30,  -1.57>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.87,   0.42,  -1.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -1.11,   1.88,  -1.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(< -1.34,  -2.44,  -7.82>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(< -0.64,  -1.76,  -7.94>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.05,  -2.07,  -8.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -4.99,  -0.81,  -4.89>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -4.18,  -0.29,  -4.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -5.70,  -0.23,  -4.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  2.83,  -1.51,  -5.17>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  2.25,  -1.28,  -5.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.99,  -0.62,  -4.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.94,  -1.38,  -1.96>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.35,  -2.25,  -2.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -0.41,  -1.77,  -1.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  2.30,   3.57,  -2.70>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  1.33,   3.74,  -2.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  2.76,   4.35,  -3.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  3.54,   1.67,  -0.77>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  3.61,   2.24,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.95,   2.19,  -1.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  2.51,   1.18,  -4.54>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(<  2.42,   1.87,  -3.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(<  3.20,   1.50,  -5.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  1.37,  -3.46,  -4.03>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  2.13,  -2.99,  -4.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  1.51,  -4.41,  -4.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  2.88,  -1.63,  -9.34>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  2.02,  -1.20,  -9.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(<  3.15,  -1.17,  -8.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -2.21,  -3.76,  -3.04>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(< -1.75,  -3.43,  -3.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(< -3.15,  -3.92,  -3.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(< -4.10,  -2.70,  -6.71>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(< -4.49,  -2.05,  -6.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -3.17,  -2.75,  -6.37>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  5.25,  -2.81,  -6.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(<  4.38,  -2.52,  -5.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(<  5.70,  -1.96,  -6.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(< -0.82,   1.98,  -6.54>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(< -1.12,   1.39,  -5.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -1.55,   2.00,  -7.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(< -1.88,   4.10,  -4.06>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(< -2.64,   4.41,  -4.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(< -1.49,   3.33,  -4.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -1.18,  -2.97,  -5.18>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(< -1.23,  -2.93,  -6.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(< -0.26,  -3.09,  -4.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -2.92,   0.55,  -4.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -3.09,   0.88,  -3.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -2.16,  -0.00,  -3.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.06,  -0.18,  -4.52>, < -0.09,  -0.50,  -4.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.23,  -0.81,  -3.82>, < -0.09,  -0.50,  -4.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.06,  -0.18,  -4.52>, <  0.37,   0.07,  -4.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,   0.33,  -4.04>, <  0.37,   0.07,  -4.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.78,   2.67,  -1.94>, < -3.20,   2.83,  -1.76>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.62,   2.98,  -1.57>, < -3.20,   2.83,  -1.76>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.78,   2.67,  -1.94>, < -2.66,   3.07,  -2.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.54,   3.47,  -2.53>, < -2.66,   3.07,  -2.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -0.80,  -7.22>, <  0.65,  -0.45,  -7.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,  -0.11,  -7.88>, <  0.65,  -0.45,  -7.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -0.80,  -7.22>, <  0.59,  -0.65,  -6.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.43,  -0.50,  -6.37>, <  0.59,  -0.65,  -6.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.41,   1.30,  -1.57>, < -0.64,   0.86,  -1.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.87,   0.42,  -1.66>, < -0.64,   0.86,  -1.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.41,   1.30,  -1.57>, < -0.76,   1.59,  -1.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.11,   1.88,  -1.85>, < -0.76,   1.59,  -1.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,  -2.44,  -7.82>, < -0.99,  -2.10,  -7.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.64,  -1.76,  -7.94>, < -0.99,  -2.10,  -7.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.34,  -2.44,  -7.82>, < -1.70,  -2.25,  -8.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.05,  -2.07,  -8.41>, < -1.70,  -2.25,  -8.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.99,  -0.81,  -4.89>, < -5.35,  -0.52,  -4.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -5.70,  -0.23,  -4.57>, < -5.35,  -0.52,  -4.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.99,  -0.81,  -4.89>, < -4.58,  -0.55,  -4.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.18,  -0.29,  -4.93>, < -4.58,  -0.55,  -4.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -1.51,  -5.17>, <  2.54,  -1.39,  -5.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.25,  -1.28,  -5.96>, <  2.54,  -1.39,  -5.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -1.51,  -5.17>, <  2.91,  -1.06,  -5.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.99,  -0.62,  -4.87>, <  2.91,  -1.06,  -5.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -1.38,  -1.96>, < -1.14,  -1.81,  -2.13>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.35,  -2.25,  -2.31>, < -1.14,  -1.81,  -2.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -1.38,  -1.96>, < -0.67,  -1.58,  -1.59>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.41,  -1.77,  -1.22>, < -0.67,  -1.58,  -1.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.30,   3.57,  -2.70>, <  2.53,   3.96,  -2.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.76,   4.35,  -3.07>, <  2.53,   3.96,  -2.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.30,   3.57,  -2.70>, <  1.81,   3.65,  -2.79>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.33,   3.74,  -2.89>, <  1.81,   3.65,  -2.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.54,   1.67,  -0.77>, <  3.58,   1.96,  -0.38>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.61,   2.24,   0.00>, <  3.58,   1.96,  -0.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.54,   1.67,  -0.77>, <  3.24,   1.93,  -1.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.95,   2.19,  -1.34>, <  3.24,   1.93,  -1.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,   1.18,  -4.54>, <  2.85,   1.34,  -4.82>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.20,   1.50,  -5.11>, <  2.85,   1.34,  -4.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,   1.18,  -4.54>, <  2.47,   1.53,  -4.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.42,   1.87,  -3.81>, <  2.47,   1.53,  -4.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.37,  -3.46,  -4.03>, <  1.44,  -3.94,  -4.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,  -4.41,  -4.30>, <  1.44,  -3.94,  -4.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.37,  -3.46,  -4.03>, <  1.75,  -3.22,  -4.22>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.13,  -2.99,  -4.41>, <  1.75,  -3.22,  -4.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,  -1.63,  -9.34>, <  3.01,  -1.40,  -8.93>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.15,  -1.17,  -8.51>, <  3.01,  -1.40,  -8.93>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,  -1.63,  -9.34>, <  2.45,  -1.42,  -9.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.02,  -1.20,  -9.57>, <  2.45,  -1.42,  -9.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,  -3.76,  -3.04>, < -1.98,  -3.60,  -3.46>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -3.43,  -3.87>, < -1.98,  -3.60,  -3.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,  -3.76,  -3.04>, < -2.68,  -3.84,  -3.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.15,  -3.92,  -3.31>, < -2.68,  -3.84,  -3.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -3.43,  -3.87>, < -1.47,  -3.20,  -4.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -2.97,  -5.18>, < -1.47,  -3.20,  -4.53>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  -2.70,  -6.71>, < -4.29,  -2.37,  -6.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.49,  -2.05,  -6.11>, < -4.29,  -2.37,  -6.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,  -2.70,  -6.71>, < -3.64,  -2.72,  -6.54>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.17,  -2.75,  -6.37>, < -3.64,  -2.72,  -6.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.25,  -2.81,  -6.00>, <  5.48,  -2.39,  -6.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  5.70,  -1.96,  -6.02>, <  5.48,  -2.39,  -6.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.25,  -2.81,  -6.00>, <  4.81,  -2.66,  -5.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.38,  -2.52,  -5.75>, <  4.81,  -2.66,  -5.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,   1.98,  -6.54>, < -1.18,   1.99,  -6.85>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.55,   2.00,  -7.17>, < -1.18,   1.99,  -6.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.82,   1.98,  -6.54>, < -0.97,   1.69,  -6.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   1.39,  -5.80>, < -0.97,   1.69,  -6.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   4.10,  -4.06>, < -2.26,   4.26,  -4.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.64,   4.41,  -4.64>, < -2.26,   4.26,  -4.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   4.10,  -4.06>, < -1.69,   3.72,  -4.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.49,   3.33,  -4.50>, < -1.69,   3.72,  -4.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -2.97,  -5.18>, < -1.21,  -2.95,  -5.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.23,  -2.93,  -6.17>, < -1.21,  -2.95,  -5.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,  -2.97,  -5.18>, < -0.72,  -3.03,  -5.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.26,  -3.09,  -4.96>, < -0.72,  -3.03,  -5.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.92,   0.55,  -4.12>, < -2.54,   0.28,  -3.97>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.16,  -0.00,  -3.83>, < -2.54,   0.28,  -3.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.92,   0.55,  -4.12>, < -3.01,   0.72,  -3.64>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.09,   0.88,  -3.16>, < -3.01,   0.72,  -3.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
