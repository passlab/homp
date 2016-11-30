set title "Offloading (axpy_kernel) Profile on 6 Devices"
set yrange [0:72.000000]
set xlabel "execution time in ms"
set xrange [0:158.400000]
set style fill pattern 2 bo 1
set style rect fs solid 1 noborder
set border 15 lw 0.2
set xtics out nomirror
unset key
set ytics out nomirror ("dev 0(sysid:0,type:HOSTCPU)" 5,"dev 1(sysid:1,type:HOSTCPU)" 15,"dev 2(sysid:0,type:THSIM)" 25,"dev 3(sysid:1,type:THSIM)" 35,"dev 4(sysid:2,type:THSIM)" 45,"dev 5(sysid:3,type:THSIM)" 55)
set object 1 rect from 4, 65 to 17, 68 fc rgb "#FF0000"
set label "ACCU_TOTAL" at 4,63 font "Helvetica,8'"

set object 2 rect from 21, 65 to 34, 68 fc rgb "#00FF00"
set label "INIT_0" at 21,63 font "Helvetica,8'"

set object 3 rect from 38, 65 to 51, 68 fc rgb "#0000FF"
set label "INIT_0.1" at 38,63 font "Helvetica,8'"

set object 4 rect from 55, 65 to 68, 68 fc rgb "#FFFF00"
set label "INIT_1" at 55,63 font "Helvetica,8'"

set object 5 rect from 72, 65 to 85, 68 fc rgb "#00FFFF"
set label "MODELING" at 72,63 font "Helvetica,8'"

set object 6 rect from 89, 65 to 102, 68 fc rgb "#FF00FF"
set label "ACC_MAPTO" at 89,63 font "Helvetica,8'"

set object 7 rect from 106, 65 to 119, 68 fc rgb "#808080"
set label "KERN" at 106,63 font "Helvetica,8'"

set object 8 rect from 123, 65 to 136, 68 fc rgb "#800000"
set label "PRE_BAR_X" at 123,63 font "Helvetica,8'"

set object 9 rect from 140, 65 to 153, 68 fc rgb "#808000"
set label "DATA_X" at 140,63 font "Helvetica,8'"

set object 10 rect from 157, 65 to 170, 68 fc rgb "#008000"
set label "POST_BAR_X" at 157,63 font "Helvetica,8'"

set object 11 rect from 174, 65 to 187, 68 fc rgb "#800080"
set label "ACC_MAPFROM" at 174,63 font "Helvetica,8'"

set object 12 rect from 191, 65 to 204, 68 fc rgb "#008080"
set label "FINI_1" at 191,63 font "Helvetica,8'"

set object 13 rect from 208, 65 to 221, 68 fc rgb "#000080"
set label "BAR_FINI_2" at 208,63 font "Helvetica,8'"

set object 14 rect from 225, 65 to 238, 68 fc rgb "(null)"
set label "PROF_BAR" at 225,63 font "Helvetica,8'"

set object 15 rect from 0.228998, 0 to 155.737664, 10 fc rgb "#FF0000"
set object 16 rect from 0.107082, 0 to 70.455782, 10 fc rgb "#00FF00"
set object 17 rect from 0.109235, 0 to 71.060909, 10 fc rgb "#0000FF"
set object 18 rect from 0.109917, 0 to 71.533372, 10 fc rgb "#FFFF00"
set object 19 rect from 0.110648, 0 to 72.201281, 10 fc rgb "#FF00FF"
set object 20 rect from 0.111691, 0 to 78.116713, 10 fc rgb "#808080"
set object 21 rect from 0.120824, 0 to 78.472030, 10 fc rgb "#800080"
set object 22 rect from 0.121628, 0 to 78.810517, 10 fc rgb "#008080"
set object 23 rect from 0.121893, 0 to 147.751187, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.227068, 10 to 154.729984, 20 fc rgb "#FF0000"
set object 25 rect from 0.082567, 10 to 54.803831, 20 fc rgb "#00FF00"
set object 26 rect from 0.085079, 10 to 55.295707, 20 fc rgb "#0000FF"
set object 27 rect from 0.085559, 10 to 55.866540, 20 fc rgb "#FFFF00"
set object 28 rect from 0.086472, 10 to 56.621169, 20 fc rgb "#FF00FF"
set object 29 rect from 0.087638, 10 to 62.657634, 20 fc rgb "#808080"
set object 30 rect from 0.096976, 10 to 63.133328, 20 fc rgb "#800080"
set object 31 rect from 0.098010, 10 to 63.591547, 20 fc rgb "#008080"
set object 32 rect from 0.098384, 10 to 146.496262, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.225021, 20 to 160.586510, 30 fc rgb "#FF0000"
set object 34 rect from 0.043062, 20 to 30.507195, 30 fc rgb "#00FF00"
set object 35 rect from 0.047614, 20 to 31.551783, 30 fc rgb "#0000FF"
set object 36 rect from 0.048886, 20 to 34.695239, 30 fc rgb "#FFFF00"
set object 37 rect from 0.053776, 20 to 39.248957, 30 fc rgb "#FF00FF"
set object 38 rect from 0.060776, 20 to 45.510650, 30 fc rgb "#808080"
set object 39 rect from 0.070461, 20 to 46.055588, 30 fc rgb "#800080"
set object 40 rect from 0.071714, 20 to 47.010216, 30 fc rgb "#008080"
set object 41 rect from 0.072786, 20 to 145.029697, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.232283, 30 to 163.624479, 40 fc rgb "#FF0000"
set object 43 rect from 0.161339, 30 to 105.333525, 40 fc rgb "#00FF00"
set object 44 rect from 0.163112, 30 to 105.779448, 40 fc rgb "#0000FF"
set object 45 rect from 0.163560, 30 to 108.043365, 40 fc rgb "#FFFF00"
set object 46 rect from 0.167086, 30 to 112.629438, 40 fc rgb "#FF00FF"
set object 47 rect from 0.174167, 30 to 118.678846, 40 fc rgb "#808080"
set object 48 rect from 0.183499, 30 to 119.124768, 40 fc rgb "#800080"
set object 49 rect from 0.184448, 30 to 119.551277, 40 fc rgb "#008080"
set object 50 rect from 0.184844, 30 to 149.996336, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.233743, 40 to 163.619319, 50 fc rgb "#FF0000"
set object 52 rect from 0.194869, 40 to 127.048472, 50 fc rgb "#00FF00"
set object 53 rect from 0.196666, 40 to 127.433560, 50 fc rgb "#0000FF"
set object 54 rect from 0.197019, 40 to 129.541493, 50 fc rgb "#FFFF00"
set object 55 rect from 0.200305, 40 to 133.671943, 50 fc rgb "#FF00FF"
set object 56 rect from 0.206659, 40 to 139.451461, 50 fc rgb "#808080"
set object 57 rect from 0.215609, 40 to 139.889620, 50 fc rgb "#800080"
set object 58 rect from 0.216563, 40 to 140.306417, 50 fc rgb "#008080"
set object 59 rect from 0.216912, 40 to 150.961312, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.230725, 50 to 161.952746, 60 fc rgb "#FF0000"
set object 61 rect from 0.130837, 50 to 85.773122, 60 fc rgb "#00FF00"
set object 62 rect from 0.132894, 50 to 86.211281, 60 fc rgb "#0000FF"
set object 63 rect from 0.133327, 50 to 88.473896, 60 fc rgb "#FFFF00"
set object 64 rect from 0.136872, 50 to 92.639941, 60 fc rgb "#FF00FF"
set object 65 rect from 0.143274, 50 to 98.458942, 60 fc rgb "#808080"
set object 66 rect from 0.152263, 50 to 98.920392, 60 fc rgb "#800080"
set object 67 rect from 0.153237, 50 to 99.445926, 60 fc rgb "#008080"
set object 68 rect from 0.153777, 50 to 148.858548, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
