#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -19.38*x up 18.84*y
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
atom(< -0.19,   5.97, -15.93>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(< -1.44,   5.09, -16.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  0.50,   7.11, -17.10>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(<  0.58,   5.25, -15.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(< -0.66,   6.54, -15.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.78,   7.88,  -8.09>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(<  2.53,   7.56,  -8.09>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(<  0.24,   8.05,  -9.82>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(<  0.64,   8.85,  -7.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<  0.17,   7.05,  -7.70>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -2.75,   1.93,  -2.98>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -3.16,   3.39,  -3.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -1.19,   1.30,  -3.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -3.53,   1.20,  -3.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.64,   2.13,  -1.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -2.45,   2.15, -14.61>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -0.66,   1.90, -14.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -2.88,   3.78, -14.02>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -3.03,   1.39, -14.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -2.74,   2.11, -15.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  2.29,   7.06, -13.14>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  3.67,   7.54, -12.06>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  2.82,   5.72, -14.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  1.92,   7.94, -13.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.49,   6.69, -12.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  7.72,   1.64, -11.19>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  6.21,   2.53, -11.20>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  8.82,   2.57, -12.34>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(<  7.52,   0.61, -11.51>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  8.05,   1.76, -10.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(<  4.29,   1.89,  -1.65>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(<  4.92,   1.50,   0.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(<  3.14,   3.22,  -1.32>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(<  3.72,   1.10,  -2.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(<  5.12,   2.22,  -2.34>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.55,   1.21,  -6.02>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(<  4.18,   1.89,  -6.38>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(<  2.67,   0.31,  -4.48>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(<  2.16,   0.59,  -6.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  1.85,   2.04,  -5.95>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  3.78,   1.50, -14.95>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(<  3.01,   2.34, -13.56>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  3.02,   1.73, -16.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  3.85,   0.40, -14.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  4.83,   1.90, -15.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(<  0.22,  -1.45, -13.44>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -0.31,  -3.10, -13.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(<  1.13,  -1.51, -14.98>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(<  0.85,  -0.98, -12.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -0.67,  -0.84, -13.61>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -1.10,   6.60,  -4.56>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -1.92,   6.28,  -6.07>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -1.90,   7.93,  -3.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -0.11,   6.93,  -4.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(< -1.07,   5.66,  -3.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -1.14,  -3.01,  -6.30>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(<  0.47,  -2.23,  -6.58>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -1.14,  -3.86,  -4.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -1.27,  -3.76,  -7.09>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -1.98,  -2.25,  -6.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(< -0.47,   1.08, -18.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  0.40,   2.02, -19.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(< -2.07,   0.52, -18.70>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(< -0.62,   1.76, -17.31>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(<  0.15,   0.19, -17.89>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  3.80,  -3.58, -13.90>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  4.16,  -2.00, -13.05>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  5.17,  -3.77, -15.08>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  2.86,  -3.49, -14.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  3.74,  -4.50, -13.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(< -1.64,   0.33,  -8.46>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(< -2.71,  -0.92,  -9.11>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(< -0.31,   0.40,  -9.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(< -1.26,   0.06,  -7.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(< -2.25,   1.28,  -8.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  2.57,   3.57,  -9.34>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  1.39,   3.59, -10.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  1.64,   3.95,  -7.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  3.29,   4.42,  -9.48>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  3.12,   2.63,  -9.27>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(<  0.75,  -1.71,  -2.10>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(<  0.04,  -3.27,  -1.42>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(<  2.20,  -1.09,  -1.21>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(<  0.00,  -0.92,  -2.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(<  0.96,  -1.92,  -3.15>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -2.22,   4.69, -10.22>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -1.79,   4.07,  -8.62>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -3.98,   4.48, -10.46>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(< -1.68,   4.19, -11.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(< -2.01,   5.78, -10.13>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(<  3.83,  -2.74,  -9.24>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(<  4.77,  -1.39,  -8.54>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(<  2.21,  -2.09,  -9.82>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(<  3.70,  -3.45,  -8.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(<  4.39,  -3.15, -10.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  7.00,   4.28,  -7.98>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  8.68,   4.21,  -8.56>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  6.91,   5.07,  -6.31>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  6.54,   3.28,  -8.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  6.46,   4.97,  -8.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {< -0.19,   5.97, -15.93>, <  0.19,   5.61, -15.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.58,   5.25, -15.55>, <  0.19,   5.61, -15.74>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   5.97, -15.93>, < -0.42,   6.25, -15.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.66,   6.54, -15.10>, < -0.42,   6.25, -15.52>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   5.97, -15.93>, < -0.82,   5.53, -16.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.44,   5.09, -16.84>, < -0.82,   5.53, -16.39>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,   5.97, -15.93>, <  0.15,   6.54, -16.52>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.50,   7.11, -17.10>, <  0.15,   6.54, -16.52>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,   7.88,  -8.09>, <  0.51,   7.96,  -8.96>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.24,   8.05,  -9.82>, <  0.51,   7.96,  -8.96>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,   7.88,  -8.09>, <  0.71,   8.36,  -7.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.64,   8.85,  -7.57>, <  0.71,   8.36,  -7.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,   7.88,  -8.09>, <  0.48,   7.46,  -7.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.17,   7.05,  -7.70>, <  0.48,   7.46,  -7.90>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.78,   7.88,  -8.09>, <  1.66,   7.72,  -8.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.53,   7.56,  -8.09>, <  1.66,   7.72,  -8.09>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,   1.93,  -2.98>, < -3.14,   1.56,  -3.11>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,   1.20,  -3.23>, < -3.14,   1.56,  -3.11>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,   1.93,  -2.98>, < -1.97,   1.62,  -3.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.19,   1.30,  -3.68>, < -1.97,   1.62,  -3.33>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,   1.93,  -2.98>, < -2.95,   2.66,  -3.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.16,   3.39,  -3.84>, < -2.95,   2.66,  -3.41>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.75,   1.93,  -2.98>, < -2.69,   2.03,  -2.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.64,   2.13,  -1.89>, < -2.69,   2.03,  -2.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   2.15, -14.61>, < -2.74,   1.77, -14.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.03,   1.39, -14.00>, < -2.74,   1.77, -14.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   2.15, -14.61>, < -1.56,   2.03, -14.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.66,   1.90, -14.55>, < -1.56,   2.03, -14.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   2.15, -14.61>, < -2.59,   2.13, -15.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.74,   2.11, -15.66>, < -2.59,   2.13, -15.13>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.45,   2.15, -14.61>, < -2.66,   2.96, -14.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,   3.78, -14.02>, < -2.66,   2.96, -14.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   7.06, -13.14>, <  2.55,   6.39, -13.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.82,   5.72, -14.15>, <  2.55,   6.39, -13.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   7.06, -13.14>, <  1.89,   6.88, -12.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.49,   6.69, -12.53>, <  1.89,   6.88, -12.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   7.06, -13.14>, <  2.10,   7.50, -13.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.92,   7.94, -13.71>, <  2.10,   7.50, -13.43>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.29,   7.06, -13.14>, <  2.98,   7.30, -12.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.67,   7.54, -12.06>, <  2.98,   7.30, -12.60>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.64, -11.19>, <  7.62,   1.13, -11.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.52,   0.61, -11.51>, <  7.62,   1.13, -11.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.64, -11.19>, <  7.88,   1.70, -10.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.05,   1.76, -10.12>, <  7.88,   1.70, -10.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.64, -11.19>, <  6.96,   2.09, -11.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.21,   2.53, -11.20>, <  6.96,   2.09, -11.19>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.72,   1.64, -11.19>, <  8.27,   2.11, -11.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.82,   2.57, -12.34>, <  8.27,   2.11, -11.76>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,   1.89,  -1.65>, <  4.01,   1.50,  -1.87>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.72,   1.10,  -2.09>, <  4.01,   1.50,  -1.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,   1.89,  -1.65>, <  4.61,   1.70,  -0.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.92,   1.50,   0.00>, <  4.61,   1.70,  -0.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,   1.89,  -1.65>, <  3.72,   2.56,  -1.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.14,   3.22,  -1.32>, <  3.72,   2.56,  -1.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.29,   1.89,  -1.65>, <  4.70,   2.06,  -1.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.12,   2.22,  -2.34>, <  4.70,   2.06,  -1.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   1.21,  -6.02>, <  2.36,   0.90,  -6.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.16,   0.59,  -6.83>, <  2.36,   0.90,  -6.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   1.21,  -6.02>, <  3.36,   1.55,  -6.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.18,   1.89,  -6.38>, <  3.36,   1.55,  -6.20>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   1.21,  -6.02>, <  2.61,   0.76,  -5.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.67,   0.31,  -4.48>, <  2.61,   0.76,  -5.25>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.55,   1.21,  -6.02>, <  2.20,   1.63,  -5.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.85,   2.04,  -5.95>, <  2.20,   1.63,  -5.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   1.50, -14.95>, <  3.81,   0.95, -14.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.85,   0.40, -14.77>, <  3.81,   0.95, -14.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   1.50, -14.95>, <  3.40,   1.62, -15.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.02,   1.73, -16.61>, <  3.40,   1.62, -15.78>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   1.50, -14.95>, <  3.39,   1.92, -14.25>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.01,   2.34, -13.56>, <  3.39,   1.92, -14.25>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.78,   1.50, -14.95>, <  4.30,   1.70, -15.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.83,   1.90, -15.08>, <  4.30,   1.70, -15.01>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,  -1.45, -13.44>, < -0.22,  -1.15, -13.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.67,  -0.84, -13.61>, < -0.22,  -1.15, -13.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,  -1.45, -13.44>, <  0.53,  -1.21, -13.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.85,  -0.98, -12.67>, <  0.53,  -1.21, -13.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,  -1.45, -13.44>, < -0.05,  -2.27, -13.28>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,  -3.10, -13.12>, < -0.05,  -2.27, -13.28>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.22,  -1.45, -13.44>, <  0.68,  -1.48, -14.21>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.13,  -1.51, -14.98>, <  0.68,  -1.48, -14.21>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.60,  -4.56>, < -1.50,   7.26,  -4.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.90,   7.93,  -3.76>, < -1.50,   7.26,  -4.16>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.60,  -4.56>, < -1.51,   6.44,  -5.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.92,   6.28,  -6.07>, < -1.51,   6.44,  -5.32>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.60,  -4.56>, < -1.09,   6.13,  -4.26>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.07,   5.66,  -3.97>, < -1.09,   6.13,  -4.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.10,   6.60,  -4.56>, < -0.60,   6.77,  -4.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.11,   6.93,  -4.77>, < -0.60,   6.77,  -4.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.01,  -6.30>, < -1.20,  -3.39,  -6.70>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.27,  -3.76,  -7.09>, < -1.20,  -3.39,  -6.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.01,  -6.30>, < -0.34,  -2.62,  -6.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.47,  -2.23,  -6.58>, < -0.34,  -2.62,  -6.44>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.01,  -6.30>, < -1.56,  -2.63,  -6.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.98,  -2.25,  -6.30>, < -1.56,  -2.63,  -6.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.01,  -6.30>, < -1.14,  -3.44,  -5.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.14,  -3.86,  -4.68>, < -1.14,  -3.44,  -5.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   1.08, -18.16>, < -0.55,   1.42, -17.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.62,   1.76, -17.31>, < -0.55,   1.42, -17.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   1.08, -18.16>, < -1.27,   0.80, -18.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.07,   0.52, -18.70>, < -1.27,   0.80, -18.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   1.08, -18.16>, < -0.16,   0.64, -18.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.15,   0.19, -17.89>, < -0.16,   0.64, -18.02>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.47,   1.08, -18.16>, < -0.04,   1.55, -18.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.40,   2.02, -19.44>, < -0.04,   1.55, -18.80>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,  -3.58, -13.90>, <  3.98,  -2.79, -13.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.16,  -2.00, -13.05>, <  3.98,  -2.79, -13.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,  -3.58, -13.90>, <  4.48,  -3.67, -14.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.17,  -3.77, -15.08>, <  4.48,  -3.67, -14.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,  -3.58, -13.90>, <  3.33,  -3.53, -14.22>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.86,  -3.49, -14.54>, <  3.33,  -3.53, -14.22>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.80,  -3.58, -13.90>, <  3.77,  -4.04, -13.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.74,  -4.50, -13.27>, <  3.77,  -4.04, -13.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,   0.33,  -8.46>, < -0.97,   0.36,  -9.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.31,   0.40,  -9.61>, < -0.97,   0.36,  -9.04>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,   0.33,  -8.46>, < -1.95,   0.80,  -8.45>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.25,   1.28,  -8.43>, < -1.95,   0.80,  -8.45>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,   0.33,  -8.46>, < -2.18,  -0.29,  -8.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.71,  -0.92,  -9.11>, < -2.18,  -0.29,  -8.79>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.64,   0.33,  -8.46>, < -1.45,   0.19,  -7.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.26,   0.06,  -7.48>, < -1.45,   0.19,  -7.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,   3.57,  -9.34>, <  1.98,   3.58, -10.01>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   3.59, -10.67>, <  1.98,   3.58, -10.01>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,   3.57,  -9.34>, <  2.93,   3.99,  -9.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.29,   4.42,  -9.48>, <  2.93,   3.99,  -9.41>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,   3.57,  -9.34>, <  2.10,   3.76,  -8.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.64,   3.95,  -7.85>, <  2.10,   3.76,  -8.60>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.57,   3.57,  -9.34>, <  2.84,   3.10,  -9.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.12,   2.63,  -9.27>, <  2.84,   3.10,  -9.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -1.71,  -2.10>, <  0.85,  -1.81,  -2.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.96,  -1.92,  -3.15>, <  0.85,  -1.81,  -2.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -1.71,  -2.10>, <  0.38,  -1.31,  -2.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.00,  -0.92,  -2.07>, <  0.38,  -1.31,  -2.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -1.71,  -2.10>, <  0.40,  -2.49,  -1.76>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.04,  -3.27,  -1.42>, <  0.40,  -2.49,  -1.76>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.75,  -1.71,  -2.10>, <  1.47,  -1.40,  -1.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.20,  -1.09,  -1.21>, <  1.47,  -1.40,  -1.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   4.69, -10.22>, < -3.10,   4.58, -10.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.98,   4.48, -10.46>, < -3.10,   4.58, -10.34>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   4.69, -10.22>, < -1.95,   4.44, -10.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.68,   4.19, -11.06>, < -1.95,   4.44, -10.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   4.69, -10.22>, < -2.12,   5.23, -10.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.01,   5.78, -10.13>, < -2.12,   5.23, -10.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.22,   4.69, -10.22>, < -2.00,   4.38,  -9.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.79,   4.07,  -8.62>, < -2.00,   4.38,  -9.42>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -2.74,  -9.24>, <  4.11,  -2.94,  -9.67>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.39,  -3.15, -10.10>, <  4.11,  -2.94,  -9.67>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -2.74,  -9.24>, <  3.02,  -2.42,  -9.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.21,  -2.09,  -9.82>, <  3.02,  -2.42,  -9.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -2.74,  -9.24>, <  3.77,  -3.10,  -8.83>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.70,  -3.45,  -8.42>, <  3.77,  -3.10,  -8.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.83,  -2.74,  -9.24>, <  4.30,  -2.07,  -8.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.77,  -1.39,  -8.54>, <  4.30,  -2.07,  -8.89>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,   4.28,  -7.98>, <  6.73,   4.63,  -8.32>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.46,   4.97,  -8.66>, <  6.73,   4.63,  -8.32>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,   4.28,  -7.98>, <  7.84,   4.25,  -8.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.68,   4.21,  -8.56>, <  7.84,   4.25,  -8.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,   4.28,  -7.98>, <  6.95,   4.68,  -7.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.91,   5.07,  -6.31>, <  6.95,   4.68,  -7.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.00,   4.28,  -7.98>, <  6.77,   3.78,  -7.99>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.54,   3.28,  -8.00>, <  6.77,   3.78,  -7.99>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
