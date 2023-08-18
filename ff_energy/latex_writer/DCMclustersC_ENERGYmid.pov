#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -17.93*x up 16.54*y
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
atom(<  3.48,  -6.85,  -9.32>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(<  5.27,  -6.64,  -9.25>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  3.11,  -7.14,  -7.63>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(<  3.34,  -7.75,  -9.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(<  2.90,  -5.95,  -9.63>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -6.33,   4.99,  -7.02>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(< -4.64,   4.72,  -7.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -6.51,   5.11,  -5.25>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(< -6.90,   4.11,  -7.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -6.70,   5.91,  -7.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(<  1.08,  -4.93, -13.03>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(<  2.57,  -5.75, -12.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(<  0.17,  -4.93, -11.52>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(<  0.56,  -5.44, -13.87>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  1.37,  -3.90, -13.35>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.00,   3.63, -16.94>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -2.37,   3.54, -15.13>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -3.24,   4.62, -17.75>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -2.15,   2.61, -17.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -0.97,   4.05, -17.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  6.76,   5.69, -14.31>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  8.13,   6.83, -13.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  5.68,   5.60, -12.93>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  7.15,   4.71, -14.60>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  6.17,   6.14, -15.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  4.48,   2.05, -12.92>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  4.00,   2.29, -14.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  5.69,   0.78, -12.81>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(<  3.55,   1.69, -12.33>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  4.91,   3.00, -12.49>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -6.15,   4.34, -13.22>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(< -4.98,   4.18, -11.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(< -7.76,   4.38, -12.41>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(< -6.05,   3.48, -13.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(< -5.98,   5.30, -13.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -5.28,  -0.73, -11.57>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< -5.38,  -2.33, -12.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(< -6.80,   0.29, -11.60>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(< -4.49,  -0.16, -12.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -4.90,  -0.96, -10.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(< -7.38,  -0.84,  -7.33>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(< -6.51,   0.41,  -6.37>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(< -8.13,  -2.12,  -6.27>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(< -6.68,  -1.41,  -8.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(< -8.17,  -0.29,  -7.92>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  1.24,  -0.50,  -8.43>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(<  2.65,   0.27,  -7.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(<  1.86,  -0.80, -10.02>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(<  0.38,   0.17,  -8.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(<  0.98,  -1.52,  -8.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -6.10,   0.07, -15.66>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -4.58,   1.02, -15.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -7.51,   1.11, -15.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -6.00,  -0.59, -14.75>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -6.10,  -0.45, -16.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -3.80,   1.21,  -8.61>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< -3.38,  -0.55,  -8.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -2.53,   2.09,  -9.46>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -3.94,   1.62,  -7.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -4.80,   1.21,  -9.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -0.71,   6.98, -11.82>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(< -1.40,   7.47, -10.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -0.83,   5.25, -11.95>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(<  0.35,   7.25, -11.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(< -1.38,   7.42, -12.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  6.03,  -0.12,  -8.90>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  6.41,   1.27,  -7.83>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  7.39,  -0.27,  -9.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  5.07,   0.06,  -9.44>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  5.90,  -1.01,  -8.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(< -1.12,   5.19,  -7.17>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(< -0.26,   6.73,  -6.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(< -0.63,   3.92,  -6.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(< -0.77,   4.87,  -8.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(< -2.21,   5.35,  -7.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  0.13,  -0.73, -13.11>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  1.81,  -0.62, -13.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(< -0.37,   0.71, -12.14>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  0.03,  -1.63, -12.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(< -0.60,  -0.88, -13.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(<  1.89,   4.97, -16.62>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(<  0.60,   6.15, -16.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(<  2.07,   4.73, -18.32>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(<  1.54,   4.05, -16.20>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(<  2.85,   5.20, -16.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(<  4.11,  -3.60,  -5.73>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(<  2.98,  -2.24,  -5.62>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(<  5.70,  -3.05,  -5.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  4.23,  -3.94,  -6.79>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(<  3.93,  -4.44,  -5.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(<  5.76,  -2.98, -13.23>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(<  5.81,  -4.70, -13.58>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(<  4.61,  -2.81, -11.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(<  5.29,  -2.50, -14.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(<  6.75,  -2.53, -13.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(< -0.78,  -6.27,  -6.73>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(< -1.03,  -4.53,  -7.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  0.33,  -6.38,  -5.27>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(< -0.29,  -6.83,  -7.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(< -1.79,  -6.65,  -6.46>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {<  3.48,  -6.85,  -9.32>, <  4.37,  -6.75,  -9.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.27,  -6.64,  -9.25>, <  4.37,  -6.75,  -9.29>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.48,  -6.85,  -9.32>, <  3.30,  -7.00,  -8.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.11,  -7.14,  -7.63>, <  3.30,  -7.00,  -8.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.48,  -6.85,  -9.32>, <  3.19,  -6.40,  -9.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.90,  -5.95,  -9.63>, <  3.19,  -6.40,  -9.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.48,  -6.85,  -9.32>, <  3.41,  -7.30,  -9.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.34,  -7.75,  -9.86>, <  3.41,  -7.30,  -9.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.33,   4.99,  -7.02>, < -6.42,   5.05,  -6.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.51,   5.11,  -5.25>, < -6.42,   5.05,  -6.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.33,   4.99,  -7.02>, < -6.52,   5.45,  -7.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.70,   5.91,  -7.51>, < -6.52,   5.45,  -7.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.33,   4.99,  -7.02>, < -5.48,   4.86,  -7.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.64,   4.72,  -7.44>, < -5.48,   4.86,  -7.23>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.33,   4.99,  -7.02>, < -6.61,   4.55,  -7.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.90,   4.11,  -7.42>, < -6.61,   4.55,  -7.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.08,  -4.93, -13.03>, <  0.63,  -4.93, -12.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.17,  -4.93, -11.52>, <  0.63,  -4.93, -12.28>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.08,  -4.93, -13.03>, <  1.23,  -4.42, -13.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.37,  -3.90, -13.35>, <  1.23,  -4.42, -13.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.08,  -4.93, -13.03>, <  1.83,  -5.34, -12.85>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,  -5.75, -12.68>, <  1.83,  -5.34, -12.85>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.08,  -4.93, -13.03>, <  0.82,  -5.19, -13.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.56,  -5.44, -13.87>, <  0.82,  -5.19, -13.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.00,   3.63, -16.94>, < -1.49,   3.84, -17.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.97,   4.05, -17.22>, < -1.49,   3.84, -17.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.00,   3.63, -16.94>, < -2.62,   4.12, -17.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.24,   4.62, -17.75>, < -2.62,   4.12, -17.35>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.00,   3.63, -16.94>, < -2.19,   3.58, -16.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.37,   3.54, -15.13>, < -2.19,   3.58, -16.04>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.00,   3.63, -16.94>, < -2.08,   3.12, -17.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.15,   2.61, -17.36>, < -2.08,   3.12, -17.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.76,   5.69, -14.31>, <  6.22,   5.64, -13.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.68,   5.60, -12.93>, <  6.22,   5.64, -13.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.76,   5.69, -14.31>, <  7.45,   6.26, -14.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.13,   6.83, -13.99>, <  7.45,   6.26, -14.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.76,   5.69, -14.31>, <  6.96,   5.20, -14.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.15,   4.71, -14.60>, <  6.96,   5.20, -14.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.76,   5.69, -14.31>, <  6.47,   5.91, -14.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.17,   6.14, -15.16>, <  6.47,   5.91, -14.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.48,   2.05, -12.92>, <  4.69,   2.52, -12.70>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.91,   3.00, -12.49>, <  4.69,   2.52, -12.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.48,   2.05, -12.92>, <  4.02,   1.87, -12.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.55,   1.69, -12.33>, <  4.02,   1.87, -12.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.48,   2.05, -12.92>, <  5.09,   1.42, -12.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.69,   0.78, -12.81>, <  5.09,   1.42, -12.87>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.48,   2.05, -12.92>, <  4.24,   2.17, -13.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.00,   2.29, -14.61>, <  4.24,   2.17, -13.77>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.15,   4.34, -13.22>, < -5.57,   4.26, -12.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.98,   4.18, -11.84>, < -5.57,   4.26, -12.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.15,   4.34, -13.22>, < -6.96,   4.36, -12.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.76,   4.38, -12.41>, < -6.96,   4.36, -12.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.15,   4.34, -13.22>, < -6.07,   4.82, -13.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.98,   5.30, -13.74>, < -6.07,   4.82, -13.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.15,   4.34, -13.22>, < -6.10,   3.91, -13.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.05,   3.48, -13.89>, < -6.10,   3.91, -13.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -0.73, -11.57>, < -5.09,  -0.84, -11.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.90,  -0.96, -10.57>, < -5.09,  -0.84, -11.07>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -0.73, -11.57>, < -6.04,  -0.22, -11.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.80,   0.29, -11.60>, < -6.04,  -0.22, -11.59>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -0.73, -11.57>, < -4.88,  -0.44, -11.84>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.49,  -0.16, -12.12>, < -4.88,  -0.44, -11.84>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.28,  -0.73, -11.57>, < -5.33,  -1.53, -12.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.38,  -2.33, -12.45>, < -5.33,  -1.53, -12.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.38,  -0.84,  -7.33>, < -6.95,  -0.21,  -6.85>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.51,   0.41,  -6.37>, < -6.95,  -0.21,  -6.85>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.38,  -0.84,  -7.33>, < -7.76,  -1.48,  -6.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.13,  -2.12,  -6.27>, < -7.76,  -1.48,  -6.80>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -7.38,  -0.84,  -7.33>, < -7.03,  -1.12,  -7.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.68,  -1.41,  -8.02>, < -7.03,  -1.12,  -7.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -7.38,  -0.84,  -7.33>, < -7.77,  -0.56,  -7.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.17,  -0.29,  -7.92>, < -7.77,  -0.56,  -7.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,  -0.50,  -8.43>, <  1.11,  -1.01,  -8.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.98,  -1.52,  -8.02>, <  1.11,  -1.01,  -8.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,  -0.50,  -8.43>, <  1.55,  -0.65,  -9.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.86,  -0.80, -10.02>, <  1.55,  -0.65,  -9.22>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,  -0.50,  -8.43>, <  0.81,  -0.17,  -8.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.38,   0.17,  -8.41>, <  0.81,  -0.17,  -8.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,  -0.50,  -8.43>, <  1.94,  -0.11,  -8.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.65,   0.27,  -7.67>, <  1.94,  -0.11,  -8.05>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   0.07, -15.66>, < -6.81,   0.59, -15.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -7.51,   1.11, -15.67>, < -6.81,   0.59, -15.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   0.07, -15.66>, < -5.34,   0.54, -15.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.58,   1.02, -15.90>, < -5.34,   0.54, -15.78>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   0.07, -15.66>, < -6.05,  -0.26, -15.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.00,  -0.59, -14.75>, < -6.05,  -0.26, -15.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,   0.07, -15.66>, < -6.10,  -0.19, -16.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.10,  -0.45, -16.65>, < -6.10,  -0.19, -16.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.80,   1.21,  -8.61>, < -3.87,   1.41,  -8.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.94,   1.62,  -7.57>, < -3.87,   1.41,  -8.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.80,   1.21,  -8.61>, < -3.59,   0.33,  -8.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.38,  -0.55,  -8.45>, < -3.59,   0.33,  -8.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.80,   1.21,  -8.61>, < -4.30,   1.21,  -8.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.80,   1.21,  -9.11>, < -4.30,   1.21,  -8.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.80,   1.21,  -8.61>, < -3.16,   1.65,  -9.03>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.53,   2.09,  -9.46>, < -3.16,   1.65,  -9.03>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.71,   6.98, -11.82>, < -1.05,   7.22, -11.03>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.40,   7.47, -10.24>, < -1.05,   7.22, -11.03>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.71,   6.98, -11.82>, < -0.77,   6.11, -11.88>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.83,   5.25, -11.95>, < -0.77,   6.11, -11.88>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.71,   6.98, -11.82>, < -1.04,   7.20, -12.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.38,   7.42, -12.59>, < -1.04,   7.20, -12.21>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.71,   6.98, -11.82>, < -0.18,   7.12, -11.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.35,   7.25, -11.93>, < -0.18,   7.12, -11.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.03,  -0.12,  -8.90>, <  5.97,  -0.57,  -8.57>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.90,  -1.01,  -8.25>, <  5.97,  -0.57,  -8.57>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.03,  -0.12,  -8.90>, <  5.55,  -0.03,  -9.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.07,   0.06,  -9.44>, <  5.55,  -0.03,  -9.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  6.03,  -0.12,  -8.90>, <  6.71,  -0.19,  -9.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.39,  -0.27,  -9.99>, <  6.71,  -0.19,  -9.45>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  6.03,  -0.12,  -8.90>, <  6.22,   0.57,  -8.37>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.41,   1.27,  -7.83>, <  6.22,   0.57,  -8.37>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   5.19,  -7.17>, < -1.67,   5.27,  -7.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.21,   5.35,  -7.14>, < -1.67,   5.27,  -7.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   5.19,  -7.17>, < -0.87,   4.56,  -6.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.63,   3.92,  -6.00>, < -0.87,   4.56,  -6.59>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   5.19,  -7.17>, < -0.69,   5.96,  -7.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.26,   6.73,  -6.85>, < -0.69,   5.96,  -7.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.12,   5.19,  -7.17>, < -0.95,   5.03,  -7.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,   4.87,  -8.19>, < -0.95,   5.03,  -7.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -0.73, -13.11>, < -0.12,  -0.01, -12.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.37,   0.71, -12.14>, < -0.12,  -0.01, -12.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -0.73, -13.11>, < -0.23,  -0.81, -13.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.60,  -0.88, -13.95>, < -0.23,  -0.81, -13.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -0.73, -13.11>, <  0.08,  -1.18, -12.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.03,  -1.63, -12.46>, <  0.08,  -1.18, -12.79>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.13,  -0.73, -13.11>, <  0.97,  -0.67, -13.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.81,  -0.62, -13.67>, <  0.97,  -0.67, -13.39>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,   4.97, -16.62>, <  2.37,   5.08, -16.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.85,   5.20, -16.16>, <  2.37,   5.08, -16.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,   4.97, -16.62>, <  1.24,   5.56, -16.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.60,   6.15, -16.28>, <  1.24,   5.56, -16.45>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,   4.97, -16.62>, <  1.98,   4.85, -17.47>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.07,   4.73, -18.32>, <  1.98,   4.85, -17.47>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,   4.97, -16.62>, <  1.71,   4.51, -16.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.54,   4.05, -16.20>, <  1.71,   4.51, -16.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,  -3.60,  -5.73>, <  4.91,  -3.32,  -5.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.70,  -3.05,  -5.23>, <  4.91,  -3.32,  -5.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,  -3.60,  -5.73>, <  4.17,  -3.77,  -6.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.23,  -3.94,  -6.79>, <  4.17,  -3.77,  -6.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,  -3.60,  -5.73>, <  3.54,  -2.92,  -5.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.98,  -2.24,  -5.62>, <  3.54,  -2.92,  -5.67>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,  -3.60,  -5.73>, <  4.02,  -4.02,  -5.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.93,  -4.44,  -5.12>, <  4.02,  -4.02,  -5.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.76,  -2.98, -13.23>, <  5.19,  -2.89, -12.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.61,  -2.81, -11.84>, <  5.19,  -2.89, -12.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.76,  -2.98, -13.23>, <  6.26,  -2.76, -13.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.75,  -2.53, -13.08>, <  6.26,  -2.76, -13.15>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.76,  -2.98, -13.23>, <  5.79,  -3.84, -13.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.81,  -4.70, -13.58>, <  5.79,  -3.84, -13.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.76,  -2.98, -13.23>, <  5.52,  -2.74, -13.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.29,  -2.50, -14.13>, <  5.52,  -2.74, -13.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -6.27,  -6.73>, < -0.54,  -6.55,  -7.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.29,  -6.83,  -7.54>, < -0.54,  -6.55,  -7.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -6.27,  -6.73>, < -0.91,  -5.40,  -7.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.03,  -4.53,  -7.28>, < -0.91,  -5.40,  -7.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -6.27,  -6.73>, < -1.28,  -6.46,  -6.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,  -6.65,  -6.46>, < -1.28,  -6.46,  -6.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.78,  -6.27,  -6.73>, < -0.22,  -6.33,  -6.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.33,  -6.38,  -5.27>, < -0.22,  -6.33,  -6.00>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
