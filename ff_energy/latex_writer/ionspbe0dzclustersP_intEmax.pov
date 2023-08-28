#include "colors.inc"
#include "finish.inc"

global_settings {assumed_gamma 1 max_trace_level 6}
background {color White transmit 1.0}
camera {orthographic angle 0
  right -7.77*x up 15.96*y
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
atom(< -3.20,  -0.70,  -5.16>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #0
atom(< -2.62,  -0.82,  -4.38>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #1
atom(< -3.53,   0.19,  -5.12>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #2
atom(< -0.45,  -4.00,  -0.86>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #3
atom(< -1.02,  -3.38,  -0.45>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #4
atom(<  0.38,  -3.52,  -0.74>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #5
atom(<  0.41,  -6.49,   0.00>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #6
atom(< -0.19,  -5.74,  -0.23>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #7
atom(< -0.05,  -7.31,  -0.25>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #8
atom(< -1.45,  -1.41,  -2.73>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #9
atom(< -0.77,  -0.72,  -2.40>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #10
atom(< -2.02,  -1.53,  -1.98>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #11
atom(<  1.31,  -1.71,  -5.12>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #12
atom(<  2.01,  -1.74,  -5.82>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #13
atom(<  0.65,  -1.13,  -5.62>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #14
atom(< -1.66,  -5.81,  -3.70>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #15
atom(< -1.83,  -6.76,  -3.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #16
atom(< -2.12,  -5.76,  -2.86>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #17
atom(<  0.65,  -7.33,  -2.98>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #18
atom(<  1.13,  -6.66,  -3.41>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #19
atom(< -0.25,  -7.05,  -3.02>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #20
atom(< -0.94,  -3.79,  -5.89>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #21
atom(< -1.71,  -4.47,  -5.85>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #22
atom(< -1.06,  -3.52,  -6.81>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #23
atom(<  1.94,  -5.18,  -4.53>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #24
atom(<  2.83,  -5.49,  -4.57>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #25
atom(<  2.00,  -4.54,  -5.29>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #26
atom(<  3.44,  -0.92,  -3.70>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #27
atom(<  3.36,  -1.16,  -2.76>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #28
atom(<  2.58,  -1.31,  -3.97>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #29
atom(< -3.18,  -5.07,  -5.91>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #30
atom(< -2.97,  -5.48,  -5.03>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #31
atom(< -3.58,  -4.22,  -5.65>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #32
atom(<  2.58,  -5.47,  -1.55>, 0.26, rgb <0.94, 0.04, 0.04>, 0.0, jmol) // #33
atom(<  1.90,  -5.81,  -0.96>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #34
atom(<  2.54,  -4.49,  -1.42>, 0.12, rgb <0.91, 0.81, 0.79>, 0.0, jmol) // #35
atom(<  0.05,  -3.55,  -3.45>, 0.81, rgb <0.50, 0.20, 0.39>, 0.0, jmol) // #36
atom(<  1.89,  -2.13,  -1.16>, 0.41, rgb <0.40, 0.89, 0.45>, 0.0, jmol) // #37
cylinder {< -3.20,  -0.70,  -5.16>, < -2.91,  -0.76,  -4.77>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.62,  -0.82,  -4.38>, < -2.91,  -0.76,  -4.77>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.20,  -0.70,  -5.16>, < -3.37,  -0.26,  -5.14>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.53,   0.19,  -5.12>, < -3.37,  -0.26,  -5.14>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,  -4.00,  -0.86>, < -0.74,  -3.69,  -0.66>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.02,  -3.38,  -0.45>, < -0.74,  -3.69,  -0.66>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,  -4.00,  -0.86>, < -0.20,  -3.77,  -2.16>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -0.20,  -3.77,  -2.16>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.45,  -4.00,  -0.86>, < -0.04,  -3.76,  -0.80>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.38,  -3.52,  -0.74>, < -0.04,  -3.76,  -0.80>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.38,  -3.52,  -0.74>, <  0.21,  -3.53,  -2.09>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, <  0.21,  -3.53,  -2.09>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,  -6.49,   0.00>, <  0.11,  -6.12,  -0.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.19,  -5.74,  -0.23>, <  0.11,  -6.12,  -0.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.41,  -6.49,   0.00>, <  0.18,  -6.90,  -0.12>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.05,  -7.31,  -0.25>, <  0.18,  -6.90,  -0.12>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,  -1.41,  -2.73>, < -1.73,  -1.47,  -2.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.02,  -1.53,  -1.98>, < -1.73,  -1.47,  -2.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,  -1.41,  -2.73>, < -1.11,  -1.07,  -2.56>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,  -0.72,  -2.40>, < -1.11,  -1.07,  -2.56>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.45,  -1.41,  -2.73>, < -0.70,  -2.48,  -3.09>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -0.70,  -2.48,  -3.09>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.77,  -0.72,  -2.40>, < -0.36,  -2.14,  -2.92>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -0.36,  -2.14,  -2.92>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -1.71,  -5.12>, <  0.68,  -2.63,  -4.29>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, <  0.68,  -2.63,  -4.29>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -1.71,  -5.12>, <  0.98,  -1.42,  -5.37>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -1.13,  -5.62>, <  0.98,  -1.42,  -5.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.31,  -1.71,  -5.12>, <  1.66,  -1.72,  -5.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.01,  -1.74,  -5.82>, <  1.66,  -1.72,  -5.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.81,  -3.70>, < -0.80,  -4.68,  -3.57>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -0.80,  -4.68,  -3.57>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.81,  -3.70>, < -1.74,  -6.29,  -3.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.83,  -6.76,  -3.97>, < -1.74,  -6.29,  -3.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.66,  -5.81,  -3.70>, < -1.89,  -5.79,  -3.28>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.12,  -5.76,  -2.86>, < -1.89,  -5.79,  -3.28>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -2.12,  -5.76,  -2.86>, < -1.03,  -4.65,  -3.16>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -1.03,  -4.65,  -3.16>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -7.33,  -2.98>, <  0.20,  -7.19,  -3.00>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -0.25,  -7.05,  -3.02>, <  0.20,  -7.19,  -3.00>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.65,  -7.33,  -2.98>, <  0.89,  -7.00,  -3.19>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.13,  -6.66,  -3.41>, <  0.89,  -7.00,  -3.19>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -3.79,  -5.89>, < -1.00,  -3.66,  -6.35>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.06,  -3.52,  -6.81>, < -1.00,  -3.66,  -6.35>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -3.79,  -5.89>, < -0.44,  -3.67,  -4.67>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -0.44,  -3.67,  -4.67>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -0.94,  -3.79,  -5.89>, < -1.32,  -4.13,  -5.87>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -4.47,  -5.85>, < -1.32,  -4.13,  -5.87>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -4.47,  -5.85>, < -0.83,  -4.01,  -4.65>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, < -0.83,  -4.01,  -4.65>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {< -1.71,  -4.47,  -5.85>, < -2.44,  -4.77,  -5.88>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -5.07,  -5.91>, < -2.44,  -4.77,  -5.88>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -5.18,  -4.53>, <  0.99,  -4.37,  -3.99>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, <  0.99,  -4.37,  -3.99>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -5.18,  -4.53>, <  1.97,  -4.86,  -4.91>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.00,  -4.54,  -5.29>, <  1.97,  -4.86,  -4.91>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  1.94,  -5.18,  -4.53>, <  2.38,  -5.34,  -4.55>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.83,  -5.49,  -4.57>, <  2.38,  -5.34,  -4.55>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.00,  -4.54,  -5.29>, <  1.03,  -4.04,  -4.37>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, <  1.03,  -4.04,  -4.37>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  3.44,  -0.92,  -3.70>, <  3.40,  -1.04,  -3.23>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  3.36,  -1.16,  -2.76>, <  3.40,  -1.04,  -3.23>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  3.44,  -0.92,  -3.70>, <  3.01,  -1.12,  -3.83>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -1.31,  -3.97>, <  3.01,  -1.12,  -3.83>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -5.07,  -5.91>, < -3.38,  -4.65,  -5.78>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -3.58,  -4.22,  -5.65>, < -3.38,  -4.65,  -5.78>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {< -3.18,  -5.07,  -5.91>, < -3.07,  -5.28,  -5.47>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {< -2.97,  -5.48,  -5.03>, < -3.07,  -5.28,  -5.47>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -5.47,  -1.55>, <  2.56,  -4.98,  -1.49>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  2.54,  -4.49,  -1.42>, <  2.56,  -4.98,  -1.49>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  2.58,  -5.47,  -1.55>, <  2.24,  -5.64,  -1.26>, Rbond texture{pigment {color rgb <0.94, 0.04, 0.04> transmit 0.0} finish{jmol}}}
cylinder {<  1.90,  -5.81,  -0.96>, <  2.24,  -5.64,  -1.26>, Rbond texture{pigment {color rgb <0.91, 0.81, 0.79> transmit 0.0} finish{jmol}}}
cylinder {<  0.05,  -3.55,  -3.45>, <  0.97,  -2.84,  -2.30>, Rbond texture{pigment {color rgb <0.50, 0.20, 0.39> transmit 0.0} finish{jmol}}}
cylinder {<  1.89,  -2.13,  -1.16>, <  0.97,  -2.84,  -2.30>, Rbond texture{pigment {color rgb <0.40, 0.89, 0.45> transmit 0.0} finish{jmol}}}
// no constraints
