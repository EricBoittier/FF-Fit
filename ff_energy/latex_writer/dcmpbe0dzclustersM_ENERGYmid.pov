#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -26.79*x up 37.64*y
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
atom(<  7.32, -15.17, -21.20>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #0
atom(<  5.67, -15.74, -21.60>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #1
atom(<  7.30, -13.95, -19.91>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #2
atom(<  7.82, -14.66, -22.06>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #3
atom(<  7.88, -16.02, -20.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<-11.08,  17.01,  -0.68>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #5
atom(<-12.35,  15.76,  -0.90>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #6
atom(< -9.64,  16.14,  -0.06>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #7
atom(<-11.42,  17.80,   0.00>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(<-10.76,  17.45,  -1.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #9
atom(< -1.98, -10.57, -18.40>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #10
atom(< -0.53, -10.64, -19.43>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #11
atom(< -3.36,  -9.79, -19.23>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #12
atom(< -1.62,  -9.99, -17.54>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(< -2.28, -11.58, -18.19>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(<  8.36,  -8.96, -25.33>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #15
atom(< 10.13,  -9.16, -25.48>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #16
atom(<  7.37, -10.29, -26.21>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #17
atom(<  8.05,  -7.96, -25.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #18
atom(<  8.21,  -9.18, -24.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(<  4.40, -13.59, -17.43>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #20
atom(<  4.16, -15.29, -17.10>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #21
atom(<  4.49, -12.85, -15.88>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #22
atom(<  3.44, -13.24, -17.93>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  5.35, -13.45, -17.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #24
atom(<  0.10, -11.09, -24.29>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #25
atom(<  0.41, -12.61, -25.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #26
atom(<  0.60, -11.44, -22.68>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #27
atom(< -0.94, -10.90, -24.30>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  0.69, -10.21, -24.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< 11.37, -12.83, -27.33>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #30
atom(< 11.48, -14.30, -28.28>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #31
atom(< 11.72, -11.49, -28.48>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #32
atom(< 12.15, -12.90, -26.55>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #33
atom(< 10.38, -12.71, -26.83>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  9.38, -11.55, -16.80>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #35
atom(< 10.58, -11.20, -15.46>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #36
atom(<  8.83, -10.05, -17.67>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
atom(<  8.45, -11.95, -16.36>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #38
atom(<  9.78, -12.29, -17.53>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #39
atom(<  3.86, -10.41, -22.08>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #40
atom(<  2.99,  -8.96, -22.72>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #41
atom(<  4.69, -10.03, -20.54>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #42
atom(<  4.70, -10.63, -22.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #43
atom(<  3.10, -11.20, -21.88>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #44
atom(< -4.12, -12.63, -22.27>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #45
atom(< -2.55, -13.24, -21.52>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #46
atom(< -4.38, -12.92, -24.03>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #47
atom(< -4.00, -11.54, -22.10>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #48
atom(< -4.98, -13.01, -21.67>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #49
atom(<  2.31,  -7.21, -27.19>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #50
atom(<  2.76,  -8.97, -26.89>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #51
atom(<  3.74,  -6.09, -26.96>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #52
atom(<  1.58,  -6.87, -26.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #53
atom(<  2.01,  -7.15, -28.24>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #54
atom(< 11.64,  -6.51, -23.16>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #55
atom(< 11.76,  -4.73, -23.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #56
atom(< 10.74,  -6.93, -21.70>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #57
atom(< 12.63,  -7.03, -23.16>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #58
atom(< 11.00,  -6.88, -23.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #59
atom(<  7.49, -16.34, -26.08>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #60
atom(<  7.47, -14.88, -25.04>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #61
atom(<  6.13, -17.52, -25.69>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #62
atom(<  8.49, -16.78, -26.01>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #63
atom(<  7.36, -15.93, -27.11>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #64
atom(<  3.70, -14.33, -24.67>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #65
atom(<  4.01, -14.39, -26.44>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #66
atom(<  4.75, -13.13, -23.92>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #67
atom(<  3.95, -15.32, -24.28>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #68
atom(<  2.65, -14.01, -24.52>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #69
atom(<  3.56,  -9.06, -16.51>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #70
atom(<  2.61, -10.33, -17.36>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #71
atom(<  3.27,  -9.34, -14.78>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #72
atom(<  3.25,  -8.05, -16.84>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #73
atom(<  4.63,  -9.37, -16.68>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #74
atom(<  5.91,  -5.89, -23.41>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #75
atom(<  7.37,  -6.99, -22.98>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #76
atom(<  6.24,  -4.73, -24.73>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #77
atom(<  5.58,  -5.39, -22.50>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #78
atom(<  5.10,  -6.56, -23.77>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #79
atom(< -3.28,  -8.30, -24.26>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #80
atom(< -2.13,  -8.91, -23.03>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #81
atom(< -4.81,  -9.16, -24.12>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #82
atom(< -2.88,  -8.54, -25.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #83
atom(< -3.42,  -7.18, -24.14>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #84
atom(<  1.07, -13.97, -15.38>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #85
atom(< -0.26, -15.17, -15.71>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #86
atom(<  1.09, -13.36, -13.64>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #87
atom(<  0.97, -13.13, -16.07>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #88
atom(<  2.04, -14.47, -15.58>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #89
atom(< 11.09, -15.07, -19.80>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #90
atom(< 10.45, -16.02, -21.17>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #91
atom(< 10.52, -15.78, -18.24>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #92
atom(< 10.64, -14.05, -19.91>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #93
atom(< 12.20, -15.06, -19.80>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #94
atom(<  9.82, -10.42, -21.74>, 0.30, rgb <0.24, 0.24, 0.25>, 0.0, jmol) // #95
atom(< 11.10, -11.46, -21.14>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #96
atom(<  8.47, -11.37, -22.50>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #97
atom(< 10.25,  -9.64, -22.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #98
atom(<  9.36,  -9.87, -20.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #99
cylinder {<  7.32, -15.17, -21.20>, <  7.57, -14.92, -21.63>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.82, -14.66, -22.06>, <  7.57, -14.92, -21.63>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.32, -15.17, -21.20>, <  7.60, -15.60, -20.98>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.88, -16.02, -20.76>, <  7.60, -15.60, -20.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.32, -15.17, -21.20>, <  7.31, -14.56, -20.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.30, -13.95, -19.91>, <  7.31, -14.56, -20.55>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.32, -15.17, -21.20>, <  6.49, -15.46, -21.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.67, -15.74, -21.60>, <  6.49, -15.46, -21.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<-11.08,  17.01,  -0.68>, <-10.92,  17.23,  -1.18>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<-10.76,  17.45,  -1.67>, <-10.92,  17.23,  -1.18>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<-11.08,  17.01,  -0.68>, <-11.25,  17.41,  -0.34>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<-11.42,  17.80,   0.00>, <-11.25,  17.41,  -0.34>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<-11.08,  17.01,  -0.68>, <-10.36,  16.57,  -0.37>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -9.64,  16.14,  -0.06>, <-10.36,  16.57,  -0.37>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<-11.08,  17.01,  -0.68>, <-11.71,  16.39,  -0.79>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<-12.35,  15.76,  -0.90>, <-11.71,  16.39,  -0.79>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.98, -10.57, -18.40>, < -2.13, -11.07, -18.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.28, -11.58, -18.19>, < -2.13, -11.07, -18.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.98, -10.57, -18.40>, < -1.26, -10.60, -18.92>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.53, -10.64, -19.43>, < -1.26, -10.60, -18.92>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.98, -10.57, -18.40>, < -2.67, -10.18, -18.81>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.36,  -9.79, -19.23>, < -2.67, -10.18, -18.81>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -1.98, -10.57, -18.40>, < -1.80, -10.28, -17.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -1.62,  -9.99, -17.54>, < -1.80, -10.28, -17.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.36,  -8.96, -25.33>, <  9.25,  -9.06, -25.41>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.13,  -9.16, -25.48>, <  9.25,  -9.06, -25.41>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.36,  -8.96, -25.33>, <  7.87,  -9.63, -25.77>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.37, -10.29, -26.21>, <  7.87,  -9.63, -25.77>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  8.36,  -8.96, -25.33>, <  8.29,  -9.07, -24.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.21,  -9.18, -24.28>, <  8.29,  -9.07, -24.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  8.36,  -8.96, -25.33>, <  8.20,  -8.46, -25.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.05,  -7.96, -25.62>, <  8.20,  -8.46, -25.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.40, -13.59, -17.43>, <  4.88, -13.52, -17.70>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.35, -13.45, -17.97>, <  4.88, -13.52, -17.70>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.40, -13.59, -17.43>, <  3.92, -13.41, -17.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.44, -13.24, -17.93>, <  3.92, -13.41, -17.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  4.40, -13.59, -17.43>, <  4.28, -14.44, -17.27>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.16, -15.29, -17.10>, <  4.28, -14.44, -17.27>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  4.40, -13.59, -17.43>, <  4.44, -13.22, -16.66>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.49, -12.85, -15.88>, <  4.44, -13.22, -16.66>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.10, -11.09, -24.29>, <  0.35, -11.26, -23.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.60, -11.44, -22.68>, <  0.35, -11.26, -23.48>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.10, -11.09, -24.29>, <  0.26, -11.85, -24.72>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.41, -12.61, -25.16>, <  0.26, -11.85, -24.72>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  0.10, -11.09, -24.29>, <  0.40, -10.65, -24.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.69, -10.21, -24.67>, <  0.40, -10.65, -24.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.10, -11.09, -24.29>, < -0.42, -10.99, -24.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.94, -10.90, -24.30>, < -0.42, -10.99, -24.30>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.37, -12.83, -27.33>, < 11.76, -12.86, -26.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 12.15, -12.90, -26.55>, < 11.76, -12.86, -26.94>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.37, -12.83, -27.33>, < 11.54, -12.16, -27.91>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 11.72, -11.49, -28.48>, < 11.54, -12.16, -27.91>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< 11.37, -12.83, -27.33>, < 11.43, -13.56, -27.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 11.48, -14.30, -28.28>, < 11.43, -13.56, -27.80>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< 11.37, -12.83, -27.33>, < 10.88, -12.77, -27.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.38, -12.71, -26.83>, < 10.88, -12.77, -27.08>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  9.38, -11.55, -16.80>, <  9.58, -11.92, -17.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  9.78, -12.29, -17.53>, <  9.58, -11.92, -17.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  9.38, -11.55, -16.80>, <  9.98, -11.37, -16.13>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.58, -11.20, -15.46>, <  9.98, -11.37, -16.13>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  9.38, -11.55, -16.80>, <  8.91, -11.75, -16.58>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.45, -11.95, -16.36>, <  8.91, -11.75, -16.58>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  9.38, -11.55, -16.80>, <  9.11, -10.80, -17.23>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.83, -10.05, -17.67>, <  9.11, -10.80, -17.23>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.86, -10.41, -22.08>, <  3.48, -10.80, -21.98>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.10, -11.20, -21.88>, <  3.48, -10.80, -21.98>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.86, -10.41, -22.08>, <  4.28, -10.22, -21.31>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.69, -10.03, -20.54>, <  4.28, -10.22, -21.31>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.86, -10.41, -22.08>, <  4.28, -10.52, -22.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.70, -10.63, -22.80>, <  4.28, -10.52, -22.44>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.86, -10.41, -22.08>, <  3.43,  -9.69, -22.40>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.99,  -8.96, -22.72>, <  3.43,  -9.69, -22.40>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.12, -12.63, -22.27>, < -4.06, -12.08, -22.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.00, -11.54, -22.10>, < -4.06, -12.08, -22.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.12, -12.63, -22.27>, < -4.25, -12.77, -23.15>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.38, -12.92, -24.03>, < -4.25, -12.77, -23.15>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -4.12, -12.63, -22.27>, < -4.55, -12.82, -21.97>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.98, -13.01, -21.67>, < -4.55, -12.82, -21.97>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -4.12, -12.63, -22.27>, < -3.33, -12.93, -21.90>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.55, -13.24, -21.52>, < -3.33, -12.93, -21.90>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.31,  -7.21, -27.19>, <  2.53,  -8.09, -27.04>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.76,  -8.97, -26.89>, <  2.53,  -8.09, -27.04>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.31,  -7.21, -27.19>, <  3.02,  -6.65, -27.08>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.74,  -6.09, -26.96>, <  3.02,  -6.65, -27.08>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  2.31,  -7.21, -27.19>, <  1.94,  -7.04, -26.82>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.58,  -6.87, -26.45>, <  1.94,  -7.04, -26.82>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.31,  -7.21, -27.19>, <  2.16,  -7.18, -27.71>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,  -7.15, -28.24>, <  2.16,  -7.18, -27.71>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.64,  -6.51, -23.16>, < 11.32,  -6.69, -23.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 11.00,  -6.88, -23.97>, < 11.32,  -6.69, -23.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.64,  -6.51, -23.16>, < 11.70,  -5.62, -23.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 11.76,  -4.73, -23.24>, < 11.70,  -5.62, -23.20>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< 11.64,  -6.51, -23.16>, < 12.14,  -6.77, -23.16>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 12.63,  -7.03, -23.16>, < 12.14,  -6.77, -23.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.64,  -6.51, -23.16>, < 11.19,  -6.72, -22.43>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.74,  -6.93, -21.70>, < 11.19,  -6.72, -22.43>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.49, -16.34, -26.08>, <  6.81, -16.93, -25.89>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.13, -17.52, -25.69>, <  6.81, -16.93, -25.89>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  7.49, -16.34, -26.08>, <  7.99, -16.56, -26.05>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.49, -16.78, -26.01>, <  7.99, -16.56, -26.05>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.49, -16.34, -26.08>, <  7.43, -16.13, -26.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.36, -15.93, -27.11>, <  7.43, -16.13, -26.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  7.49, -16.34, -26.08>, <  7.48, -15.61, -25.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.47, -14.88, -25.04>, <  7.48, -15.61, -25.56>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.70, -14.33, -24.67>, <  3.82, -14.82, -24.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.95, -15.32, -24.28>, <  3.82, -14.82, -24.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.70, -14.33, -24.67>, <  3.85, -14.36, -25.56>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.01, -14.39, -26.44>, <  3.85, -14.36, -25.56>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.70, -14.33, -24.67>, <  4.22, -13.73, -24.30>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.75, -13.13, -23.92>, <  4.22, -13.73, -24.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.70, -14.33, -24.67>, <  3.17, -14.17, -24.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.65, -14.01, -24.52>, <  3.17, -14.17, -24.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -9.06, -16.51>, <  4.10,  -9.21, -16.60>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  4.63,  -9.37, -16.68>, <  4.10,  -9.21, -16.60>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -9.06, -16.51>, <  3.41,  -9.20, -15.65>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.27,  -9.34, -14.78>, <  3.41,  -9.20, -15.65>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -9.06, -16.51>, <  3.08,  -9.70, -16.94>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.61, -10.33, -17.36>, <  3.08,  -9.70, -16.94>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  3.56,  -9.06, -16.51>, <  3.41,  -8.56, -16.68>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  3.25,  -8.05, -16.84>, <  3.41,  -8.56, -16.68>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.91,  -5.89, -23.41>, <  5.50,  -6.23, -23.59>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.10,  -6.56, -23.77>, <  5.50,  -6.23, -23.59>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.91,  -5.89, -23.41>, <  6.08,  -5.31, -24.07>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  6.24,  -4.73, -24.73>, <  6.08,  -5.31, -24.07>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  5.91,  -5.89, -23.41>, <  5.75,  -5.64, -22.95>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  5.58,  -5.39, -22.50>, <  5.75,  -5.64, -22.95>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  5.91,  -5.89, -23.41>, <  6.64,  -6.44, -23.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  7.37,  -6.99, -22.98>, <  6.64,  -6.44, -23.20>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -8.30, -24.26>, < -4.04,  -8.73, -24.19>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -4.81,  -9.16, -24.12>, < -4.04,  -8.73, -24.19>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -8.30, -24.26>, < -3.08,  -8.42, -24.78>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.88,  -8.54, -25.29>, < -3.08,  -8.42, -24.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -8.30, -24.26>, < -3.35,  -7.74, -24.20>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -3.42,  -7.18, -24.14>, < -3.35,  -7.74, -24.20>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.28,  -8.30, -24.26>, < -2.70,  -8.61, -23.64>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -2.13,  -8.91, -23.03>, < -2.70,  -8.61, -23.64>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.07, -13.97, -15.38>, <  1.08, -13.66, -14.51>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  1.09, -13.36, -13.64>, <  1.08, -13.66, -14.51>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.07, -13.97, -15.38>, <  0.41, -14.57, -15.55>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< -0.26, -15.17, -15.71>, <  0.41, -14.57, -15.55>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  1.07, -13.97, -15.38>, <  1.02, -13.55, -15.73>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  0.97, -13.13, -16.07>, <  1.02, -13.55, -15.73>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.07, -13.97, -15.38>, <  1.56, -14.22, -15.48>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  2.04, -14.47, -15.58>, <  1.56, -14.22, -15.48>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.09, -15.07, -19.80>, < 10.77, -15.54, -20.49>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.45, -16.02, -21.17>, < 10.77, -15.54, -20.49>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< 11.09, -15.07, -19.80>, < 10.87, -14.56, -19.86>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.64, -14.05, -19.91>, < 10.87, -14.56, -19.86>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< 11.09, -15.07, -19.80>, < 10.80, -15.43, -19.02>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.52, -15.78, -18.24>, < 10.80, -15.43, -19.02>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {< 11.09, -15.07, -19.80>, < 11.64, -15.06, -19.80>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 12.20, -15.06, -19.80>, < 11.64, -15.06, -19.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  9.82, -10.42, -21.74>, <  9.15, -10.89, -22.12>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  8.47, -11.37, -22.50>, <  9.15, -10.89, -22.12>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  9.82, -10.42, -21.74>, < 10.46, -10.94, -21.44>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 11.10, -11.46, -21.14>, < 10.46, -10.94, -21.44>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
cylinder {<  9.82, -10.42, -21.74>, <  9.59, -10.15, -21.29>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {<  9.36,  -9.87, -20.85>, <  9.59, -10.15, -21.29>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  9.82, -10.42, -21.74>, < 10.04, -10.03, -22.06>, Rbond texture{pigment {color rgb <0.24, 0.24, 0.25> transmit 0.0} finish{jmol}}}
cylinder {< 10.25,  -9.64, -22.38>, < 10.04, -10.03, -22.06>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
// no constraints
