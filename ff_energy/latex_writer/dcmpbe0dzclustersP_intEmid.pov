#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -17.98*x up 17.99*y
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
atom(<  4.52,   3.06,  -8.12>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(<  4.93,   1.37,  -7.82>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  2.93,   3.43,  -7.36>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(<  4.54,   3.22,  -9.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(<  5.22,   3.76,  -7.59>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(< -2.07,  -3.35, -12.20>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(< -1.52,  -1.72, -11.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -3.83,  -3.90, -11.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(< -1.48,  -4.15, -11.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -1.75,  -3.36, -13.26>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(<  3.41,   2.38, -17.74>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(<  2.79,   3.96, -18.17>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(<  2.72,   1.00, -18.57>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(<  3.17,   2.23, -16.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  4.49,   2.34, -17.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -6.66,   3.48, -15.68>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< -8.15,   2.93, -15.01>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(< -6.76,   3.26, -17.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(< -5.82,   2.84, -15.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(< -6.48,   4.53, -15.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -4.53,   0.43, -11.69>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(< -4.23,   0.74,  -9.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(< -6.05,  -0.56, -11.93>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(< -3.70,  -0.19, -12.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(< -4.65,   1.38, -12.22>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  7.65,   5.90,  -6.62>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  7.45,   4.22,  -6.10>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  6.08,   6.66,  -6.45>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(<  7.89,   5.81,  -7.71>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  8.44,   6.41,  -6.04>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -2.09,   5.92, -15.47>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(< -3.58,   6.54, -16.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(< -2.53,   4.79, -14.18>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(< -1.49,   6.74, -15.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(< -1.50,   5.34, -16.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(< -5.92,   2.48,  -7.92>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< -6.21,   2.99,  -6.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(< -6.58,   3.61,  -9.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(< -6.31,   1.48,  -8.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(< -4.81,   2.60,  -8.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  5.83,  -1.03, -12.27>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(<  4.04,  -1.23, -12.30>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  6.45,  -0.49, -10.69>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  6.20,  -0.31, -13.01>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  6.21,  -2.02, -12.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -4.43,   1.13,  -2.31>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -6.06,   0.87,  -2.99>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -4.54,   1.96,  -0.74>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -3.78,   1.74,  -2.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -3.98,   0.11,  -2.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(< -0.76,  -7.37, -11.19>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(< -1.32,  -6.11,  -9.97>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(< -1.57,  -7.01, -12.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(< -0.96,  -8.44, -10.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  0.32,  -7.26, -11.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< -4.06,   5.61,  -4.61>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< -3.60,   5.17,  -6.27>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< -3.19,   4.49,  -3.55>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< -5.17,   5.49,  -4.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< -3.71,   6.65,  -4.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(<  1.39,   5.35, -12.00>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  3.00,   5.53, -11.15>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(<  0.32,   6.83, -11.84>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(<  0.87,   4.46, -11.64>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(<  1.64,   5.19, -13.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  3.57,  -1.58,  -5.18>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  5.20,  -2.20,  -5.65>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  2.74,  -2.94,  -4.30>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  2.97,  -1.33,  -6.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  3.73,  -0.74,  -4.43>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  1.83,  -4.63,  -9.60>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(<  2.06,  -5.15, -11.32>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  3.17,  -5.28,  -8.65>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  0.90,  -5.01,  -9.17>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  1.91,  -3.55,  -9.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  5.29,   3.15, -13.48>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  5.30,   2.57, -11.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  3.73,   2.99, -14.34>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  6.08,   2.61, -14.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  5.53,   4.23, -13.39>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(<  4.11,   0.77,  -1.24>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(<  2.88,   0.40,   0.00>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(<  4.72,  -0.87,  -1.61>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(<  4.96,   1.37,  -0.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(<  3.60,   1.27,  -2.08>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(< -0.45,   1.43,  -9.01>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -0.68,   3.00,  -9.87>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(< -1.70,   1.18,  -7.76>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  0.55,   1.41,  -8.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(< -0.62,   0.61,  -9.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(<  0.16,   1.89,  -4.49>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(<  0.92,   3.33,  -3.85>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(<  1.24,   0.56,  -4.05>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< -0.03,   2.12,  -5.56>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< -0.76,   1.82,  -3.99>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  2.40,   7.33,  -6.93>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(<  2.74,   6.89,  -5.26>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  0.69,   6.82,  -7.34>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(<  3.05,   6.80,  -7.66>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  2.51,   8.44,  -7.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {<  4.52,   3.06,  -8.12>, <  4.53,   3.14,  -8.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.54,   3.22,  -9.19>, <  4.53,   3.14,  -8.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.52,   3.06,  -8.12>, <  3.73,   3.25,  -7.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.93,   3.43,  -7.36>, <  3.73,   3.25,  -7.74>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.52,   3.06,  -8.12>, <  4.87,   3.41,  -7.85>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.22,   3.76,  -7.59>, <  4.87,   3.41,  -7.85>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.52,   3.06,  -8.12>, <  4.73,   2.22,  -7.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.93,   1.37,  -7.82>, <  4.73,   2.22,  -7.97>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.07,  -3.35, -12.20>, < -1.91,  -3.35, -12.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.75,  -3.36, -13.26>, < -1.91,  -3.35, -12.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.07,  -3.35, -12.20>, < -2.95,  -3.63, -12.10>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.83,  -3.90, -11.99>, < -2.95,  -3.63, -12.10>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.07,  -3.35, -12.20>, < -1.78,  -3.75, -11.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.48,  -4.15, -11.71>, < -1.78,  -3.75, -11.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.07,  -3.35, -12.20>, < -1.80,  -2.53, -11.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.52,  -1.72, -11.45>, < -1.80,  -2.53, -11.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.41,   2.38, -17.74>, <  3.10,   3.17, -17.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.79,   3.96, -18.17>, <  3.10,   3.17, -17.95>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.41,   2.38, -17.74>, <  3.29,   2.31, -17.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.17,   2.23, -16.67>, <  3.29,   2.31, -17.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.41,   2.38, -17.74>, <  3.95,   2.36, -17.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.49,   2.34, -17.91>, <  3.95,   2.36, -17.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.41,   2.38, -17.74>, <  3.06,   1.69, -18.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.72,   1.00, -18.57>, <  3.06,   1.69, -18.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.66,   3.48, -15.68>, < -6.24,   3.16, -15.46>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.82,   2.84, -15.24>, < -6.24,   3.16, -15.46>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.66,   3.48, -15.68>, < -7.40,   3.21, -15.35>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -8.15,   2.93, -15.01>, < -7.40,   3.21, -15.35>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -6.66,   3.48, -15.68>, < -6.57,   4.01, -15.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.48,   4.53, -15.40>, < -6.57,   4.01, -15.54>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -6.66,   3.48, -15.68>, < -6.71,   3.37, -16.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.76,   3.26, -17.43>, < -6.71,   3.37, -16.56>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,   0.43, -11.69>, < -4.59,   0.90, -11.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.65,   1.38, -12.22>, < -4.59,   0.90, -11.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,   0.43, -11.69>, < -4.38,   0.58, -10.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.23,   0.74,  -9.90>, < -4.38,   0.58, -10.79>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,   0.43, -11.69>, < -4.11,   0.12, -11.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.70,  -0.19, -12.16>, < -4.11,   0.12, -11.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.53,   0.43, -11.69>, < -5.29,  -0.07, -11.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.05,  -0.56, -11.93>, < -5.29,  -0.07, -11.81>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.65,   5.90,  -6.62>, <  8.04,   6.16,  -6.33>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.44,   6.41,  -6.04>, <  8.04,   6.16,  -6.33>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.65,   5.90,  -6.62>, <  7.77,   5.86,  -7.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.89,   5.81,  -7.71>, <  7.77,   5.86,  -7.17>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.65,   5.90,  -6.62>, <  6.87,   6.28,  -6.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.08,   6.66,  -6.45>, <  6.87,   6.28,  -6.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.65,   5.90,  -6.62>, <  7.55,   5.06,  -6.36>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.45,   4.22,  -6.10>, <  7.55,   5.06,  -6.36>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.09,   5.92, -15.47>, < -1.79,   6.33, -15.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.49,   6.74, -15.07>, < -1.79,   6.33, -15.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.09,   5.92, -15.47>, < -1.79,   5.63, -15.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.50,   5.34, -16.25>, < -1.79,   5.63, -15.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.09,   5.92, -15.47>, < -2.31,   5.36, -14.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.53,   4.79, -14.18>, < -2.31,   5.36, -14.82>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -2.09,   5.92, -15.47>, < -2.83,   6.23, -15.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.58,   6.54, -16.26>, < -2.83,   6.23, -15.86>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.92,   2.48,  -7.92>, < -6.25,   3.05,  -8.54>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.58,   3.61,  -9.16>, < -6.25,   3.05,  -8.54>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.92,   2.48,  -7.92>, < -6.06,   2.74,  -7.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.21,   2.99,  -6.24>, < -6.06,   2.74,  -7.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -5.92,   2.48,  -7.92>, < -5.37,   2.54,  -7.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.81,   2.60,  -8.02>, < -5.37,   2.54,  -7.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -5.92,   2.48,  -7.92>, < -6.11,   1.98,  -8.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.31,   1.48,  -8.07>, < -6.11,   1.98,  -8.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.83,  -1.03, -12.27>, <  6.02,  -0.67, -12.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.20,  -0.31, -13.01>, <  6.02,  -0.67, -12.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.83,  -1.03, -12.27>, <  6.14,  -0.76, -11.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.45,  -0.49, -10.69>, <  6.14,  -0.76, -11.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.83,  -1.03, -12.27>, <  4.94,  -1.13, -12.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.04,  -1.23, -12.30>, <  4.94,  -1.13, -12.29>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.83,  -1.03, -12.27>, <  6.02,  -1.52, -12.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.21,  -2.02, -12.56>, <  6.02,  -1.52, -12.42>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.43,   1.13,  -2.31>, < -4.48,   1.55,  -1.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.54,   1.96,  -0.74>, < -4.48,   1.55,  -1.53>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.43,   1.13,  -2.31>, < -4.11,   1.44,  -2.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.78,   1.74,  -2.98>, < -4.11,   1.44,  -2.64>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.43,   1.13,  -2.31>, < -5.24,   1.00,  -2.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -6.06,   0.87,  -2.99>, < -5.24,   1.00,  -2.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.43,   1.13,  -2.31>, < -4.20,   0.62,  -2.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.98,   0.11,  -2.23>, < -4.20,   0.62,  -2.27>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.76,  -7.37, -11.19>, < -1.04,  -6.74, -10.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.32,  -6.11,  -9.97>, < -1.04,  -6.74, -10.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.76,  -7.37, -11.19>, < -1.17,  -7.19, -11.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.57,  -7.01, -12.71>, < -1.17,  -7.19, -11.95>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.76,  -7.37, -11.19>, < -0.86,  -7.90, -11.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.96,  -8.44, -10.91>, < -0.86,  -7.90, -11.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.76,  -7.37, -11.19>, < -0.22,  -7.31, -11.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.32,  -7.26, -11.43>, < -0.22,  -7.31, -11.31>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.06,   5.61,  -4.61>, < -3.62,   5.05,  -4.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.19,   4.49,  -3.55>, < -3.62,   5.05,  -4.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.06,   5.61,  -4.61>, < -3.83,   5.39,  -5.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.60,   5.17,  -6.27>, < -3.83,   5.39,  -5.44>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.06,   5.61,  -4.61>, < -4.61,   5.55,  -4.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -5.17,   5.49,  -4.56>, < -4.61,   5.55,  -4.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.06,   5.61,  -4.61>, < -3.89,   6.13,  -4.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.71,   6.65,  -4.41>, < -3.89,   6.13,  -4.51>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   5.35, -12.00>, <  0.85,   6.09, -11.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.32,   6.83, -11.84>, <  0.85,   6.09, -11.92>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   5.35, -12.00>, <  1.13,   4.91, -11.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.87,   4.46, -11.64>, <  1.13,   4.91, -11.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   5.35, -12.00>, <  1.51,   5.27, -12.53>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.64,   5.19, -13.06>, <  1.51,   5.27, -12.53>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.39,   5.35, -12.00>, <  2.19,   5.44, -11.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.00,   5.53, -11.15>, <  2.19,   5.44, -11.58>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.57,  -1.58,  -5.18>, <  4.38,  -1.89,  -5.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.20,  -2.20,  -5.65>, <  4.38,  -1.89,  -5.42>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.57,  -1.58,  -5.18>, <  3.15,  -2.26,  -4.74>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.74,  -2.94,  -4.30>, <  3.15,  -2.26,  -4.74>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.57,  -1.58,  -5.18>, <  3.27,  -1.46,  -5.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.97,  -1.33,  -6.06>, <  3.27,  -1.46,  -5.62>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.57,  -1.58,  -5.18>, <  3.65,  -1.16,  -4.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.73,  -0.74,  -4.43>, <  3.65,  -1.16,  -4.81>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.83,  -4.63,  -9.60>, <  1.87,  -4.09,  -9.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.91,  -3.55,  -9.58>, <  1.87,  -4.09,  -9.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.83,  -4.63,  -9.60>, <  1.95,  -4.89, -10.46>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.06,  -5.15, -11.32>, <  1.95,  -4.89, -10.46>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.83,  -4.63,  -9.60>, <  1.36,  -4.82,  -9.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.90,  -5.01,  -9.17>, <  1.36,  -4.82,  -9.39>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.83,  -4.63,  -9.60>, <  2.50,  -4.95,  -9.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.17,  -5.28,  -8.65>, <  2.50,  -4.95,  -9.13>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.29,   3.15, -13.48>, <  4.51,   3.07, -13.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.73,   2.99, -14.34>, <  4.51,   3.07, -13.91>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.29,   3.15, -13.48>, <  5.68,   2.88, -13.75>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.08,   2.61, -14.02>, <  5.68,   2.88, -13.75>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.29,   3.15, -13.48>, <  5.41,   3.69, -13.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.53,   4.23, -13.39>, <  5.41,   3.69, -13.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.29,   3.15, -13.48>, <  5.29,   2.86, -12.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.30,   2.57, -11.78>, <  5.29,   2.86, -12.63>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,   0.77,  -1.24>, <  4.53,   1.07,  -1.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.96,   1.37,  -0.76>, <  4.53,   1.07,  -1.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,   0.77,  -1.24>, <  3.85,   1.02,  -1.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.60,   1.27,  -2.08>, <  3.85,   1.02,  -1.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,   0.77,  -1.24>, <  4.41,  -0.05,  -1.42>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.72,  -0.87,  -1.61>, <  4.41,  -0.05,  -1.42>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.11,   0.77,  -1.24>, <  3.49,   0.58,  -0.62>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.88,   0.40,   0.00>, <  3.49,   0.58,  -0.62>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   1.43,  -9.01>, < -0.57,   2.21,  -9.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.68,   3.00,  -9.87>, < -0.57,   2.21,  -9.44>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   1.43,  -9.01>, < -0.54,   1.02,  -9.38>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.62,   0.61,  -9.76>, < -0.54,   1.02,  -9.38>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   1.43,  -9.01>, < -1.08,   1.30,  -8.39>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.70,   1.18,  -7.76>, < -1.08,   1.30,  -8.39>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,   1.43,  -9.01>, <  0.05,   1.42,  -8.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.55,   1.41,  -8.53>, <  0.05,   1.42,  -8.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.16,   1.89,  -4.49>, <  0.54,   2.61,  -4.17>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.92,   3.33,  -3.85>, <  0.54,   2.61,  -4.17>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.16,   1.89,  -4.49>, <  0.07,   2.01,  -5.03>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.03,   2.12,  -5.56>, <  0.07,   2.01,  -5.03>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.16,   1.89,  -4.49>, < -0.30,   1.86,  -4.24>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.76,   1.82,  -3.99>, < -0.30,   1.86,  -4.24>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.16,   1.89,  -4.49>, <  0.70,   1.23,  -4.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.24,   0.56,  -4.05>, <  0.70,   1.23,  -4.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.40,   7.33,  -6.93>, <  1.54,   7.08,  -7.14>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.69,   6.82,  -7.34>, <  1.54,   7.08,  -7.14>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.40,   7.33,  -6.93>, <  2.45,   7.89,  -7.00>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.51,   8.44,  -7.07>, <  2.45,   7.89,  -7.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.40,   7.33,  -6.93>, <  2.73,   7.07,  -7.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.05,   6.80,  -7.66>, <  2.73,   7.07,  -7.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.40,   7.33,  -6.93>, <  2.57,   7.11,  -6.09>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.74,   6.89,  -5.26>, <  2.57,   7.11,  -6.09>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
