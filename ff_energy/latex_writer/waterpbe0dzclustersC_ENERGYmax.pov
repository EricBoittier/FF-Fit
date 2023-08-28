#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -9.53*x up 10.31*y
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
atom(<  0.18,   0.27,  -5.76>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(<  0.81,   0.97,  -5.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -0.33,   0.75,  -6.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -2.67,   2.84,  -2.93>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -2.98,   3.22,  -2.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -3.47,   2.25,  -3.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  0.80,  -2.43,  -6.46>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(<  0.11,  -3.09,  -6.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(<  0.50,  -1.59,  -6.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -1.30,   3.21,  -5.29>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.42,   3.13,  -4.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -1.88,   2.94,  -4.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  2.44,  -3.03,  -4.36>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  1.76,  -3.15,  -5.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  2.07,  -3.63,  -3.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  0.81,  -0.10,  -0.72>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(<  0.29,  -0.14,  -1.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(<  0.23,   0.22,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  1.51,  -0.38,  -8.53>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  1.28,  -1.18,  -7.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  0.62,  -0.09,  -8.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(<  1.38,   2.83,  -4.76>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(<  2.25,   2.48,  -4.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(<  1.77,   3.57,  -5.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -3.57,   2.09,  -8.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(< -3.74,   1.90,  -7.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(< -4.35,   1.72,  -8.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(< -1.67,   4.02,  -9.24>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(< -2.24,   4.79,  -9.32>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(< -2.19,   3.45,  -8.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -1.18,   0.25,  -8.21>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -1.87,   0.98,  -8.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -1.50,  -0.32,  -8.90>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  3.46,   0.76,  -7.08>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  2.99,   0.37,  -7.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  4.41,   0.49,  -7.21>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  1.63,  -4.55,  -8.58>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #36
atom(<  1.20,  -4.79,  -7.72>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #37
atom(<  1.17,  -3.72,  -8.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  3.76,   1.75,  -3.50>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #39
atom(<  3.43,   0.92,  -3.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #40
atom(<  3.92,   1.48,  -2.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #41
atom(<  2.70,  -1.92,  -1.60>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #42
atom(<  2.05,  -1.32,  -1.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  2.75,  -1.59,  -2.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  2.81,  -1.21, -10.95>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #45
atom(<  2.52,  -0.97, -10.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #46
atom(<  1.96,  -1.23, -11.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #47
atom(< -1.98,  -1.75,  -4.88>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #48
atom(< -1.39,  -1.01,  -4.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -2.00,  -2.39,  -4.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #50
atom(< -4.10,   0.72,  -3.82>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #51
atom(< -4.41,   0.14,  -3.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #52
atom(< -3.33,   0.29,  -4.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  2.97,  -0.39,  -4.52>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #54
atom(<  2.93,  -1.38,  -4.47>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #55
atom(<  3.09,  -0.17,  -5.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #56
atom(< -0.55,   0.67,  -2.78>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #57
atom(< -0.88,   1.63,  -2.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(<  0.00,   0.96,  -3.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
cylinder {<  0.18,   0.27,  -5.76>, <  0.50,   0.62,  -5.62>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.81,   0.97,  -5.47>, <  0.50,   0.62,  -5.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.18,   0.27,  -5.76>, < -0.07,   0.51,  -6.08>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.33,   0.75,  -6.40>, < -0.07,   0.51,  -6.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.67,   2.84,  -2.93>, < -2.83,   3.03,  -2.51>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.98,   3.22,  -2.09>, < -2.83,   3.03,  -2.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.67,   2.84,  -2.93>, < -3.07,   2.55,  -3.05>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.47,   2.25,  -3.17>, < -3.07,   2.55,  -3.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.80,  -2.43,  -6.46>, <  0.45,  -2.76,  -6.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.11,  -3.09,  -6.26>, <  0.45,  -2.76,  -6.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.80,  -2.43,  -6.46>, <  0.65,  -2.01,  -6.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,  -1.59,  -6.13>, <  0.65,  -2.01,  -6.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.30,   3.21,  -5.29>, < -0.86,   3.17,  -5.06>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.42,   3.13,  -4.83>, < -0.86,   3.17,  -5.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.30,   3.21,  -5.29>, < -1.59,   3.08,  -4.94>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.88,   2.94,  -4.59>, < -1.59,   3.08,  -4.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.44,  -3.03,  -4.36>, <  2.10,  -3.09,  -4.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.76,  -3.15,  -5.06>, <  2.10,  -3.09,  -4.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.44,  -3.03,  -4.36>, <  2.26,  -3.33,  -4.01>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.07,  -3.63,  -3.65>, <  2.26,  -3.33,  -4.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.81,  -0.10,  -0.72>, <  0.52,   0.06,  -0.36>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.23,   0.22,   0.00>, <  0.52,   0.06,  -0.36>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.81,  -0.10,  -0.72>, <  0.55,  -0.12,  -1.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.29,  -0.14,  -1.53>, <  0.55,  -0.12,  -1.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,  -0.38,  -8.53>, <  1.39,  -0.78,  -8.24>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.28,  -1.18,  -7.96>, <  1.39,  -0.78,  -8.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.51,  -0.38,  -8.53>, <  1.06,  -0.24,  -8.61>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.62,  -0.09,  -8.70>, <  1.06,  -0.24,  -8.61>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.38,   2.83,  -4.76>, <  1.81,   2.65,  -4.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.25,   2.48,  -4.36>, <  1.81,   2.65,  -4.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.38,   2.83,  -4.76>, <  1.58,   3.20,  -5.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.77,   3.57,  -5.29>, <  1.58,   3.20,  -5.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.57,   2.09,  -8.00>, < -3.96,   1.90,  -8.23>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.35,   1.72,  -8.47>, < -3.96,   1.90,  -8.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.57,   2.09,  -8.00>, < -3.65,   1.99,  -7.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.74,   1.90,  -7.04>, < -3.65,   1.99,  -7.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.67,   4.02,  -9.24>, < -1.96,   4.40,  -9.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.24,   4.79,  -9.32>, < -1.96,   4.40,  -9.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.67,   4.02,  -9.24>, < -1.93,   3.74,  -8.95>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.19,   3.45,  -8.65>, < -1.93,   3.74,  -8.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,   0.25,  -8.21>, < -1.34,  -0.04,  -8.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.50,  -0.32,  -8.90>, < -1.34,  -0.04,  -8.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.18,   0.25,  -8.21>, < -1.53,   0.61,  -8.17>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.87,   0.98,  -8.13>, < -1.53,   0.61,  -8.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.46,   0.76,  -7.08>, <  3.94,   0.63,  -7.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  4.41,   0.49,  -7.21>, <  3.94,   0.63,  -7.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.46,   0.76,  -7.08>, <  3.23,   0.57,  -7.45>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.99,   0.37,  -7.83>, <  3.23,   0.57,  -7.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.63,  -4.55,  -8.58>, <  1.40,  -4.14,  -8.71>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.17,  -3.72,  -8.85>, <  1.40,  -4.14,  -8.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.63,  -4.55,  -8.58>, <  1.42,  -4.67,  -8.15>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.20,  -4.79,  -7.72>, <  1.42,  -4.67,  -8.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.76,   1.75,  -3.50>, <  3.60,   1.33,  -3.72>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.43,   0.92,  -3.95>, <  3.60,   1.33,  -3.72>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.76,   1.75,  -3.50>, <  3.84,   1.61,  -3.03>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.92,   1.48,  -2.55>, <  3.84,   1.61,  -3.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.43,   0.92,  -3.95>, <  3.20,   0.26,  -4.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.97,  -0.39,  -4.52>, <  3.20,   0.26,  -4.23>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.70,  -1.92,  -1.60>, <  2.37,  -1.62,  -1.41>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.05,  -1.32,  -1.22>, <  2.37,  -1.62,  -1.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.70,  -1.92,  -1.60>, <  2.72,  -1.75,  -2.07>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.75,  -1.59,  -2.53>, <  2.72,  -1.75,  -2.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.81,  -1.21, -10.95>, <  2.39,  -1.22, -11.18>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.96,  -1.23, -11.41>, <  2.39,  -1.22, -11.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.81,  -1.21, -10.95>, <  2.67,  -1.09, -10.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.52,  -0.97, -10.09>, <  2.67,  -1.09, -10.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.98,  -1.75,  -4.88>, < -1.99,  -2.07,  -4.52>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.00,  -2.39,  -4.16>, < -1.99,  -2.07,  -4.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.98,  -1.75,  -4.88>, < -1.69,  -1.38,  -4.73>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.39,  -1.01,  -4.58>, < -1.69,  -1.38,  -4.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,   0.72,  -3.82>, < -4.26,   0.43,  -3.44>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -4.41,   0.14,  -3.06>, < -4.26,   0.43,  -3.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.10,   0.72,  -3.82>, < -3.72,   0.51,  -4.02>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.33,   0.29,  -4.22>, < -3.72,   0.51,  -4.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.97,  -0.39,  -4.52>, <  2.95,  -0.89,  -4.50>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,  -1.38,  -4.47>, <  2.95,  -0.89,  -4.50>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.97,  -0.39,  -4.52>, <  3.03,  -0.28,  -4.98>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.09,  -0.17,  -5.45>, <  3.03,  -0.28,  -4.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.55,   0.67,  -2.78>, < -0.28,   0.81,  -3.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,   0.96,  -3.54>, < -0.28,   0.81,  -3.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.55,   0.67,  -2.78>, < -0.72,   1.15,  -2.70>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.88,   1.63,  -2.61>, < -0.72,   1.15,  -2.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
