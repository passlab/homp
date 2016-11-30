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

set object 15 rect from 1.447343, 0 to 156.329067, 10 fc rgb "#FF0000"
set object 16 rect from 0.956589, 0 to 101.060082, 10 fc rgb "#00FF00"
set object 17 rect from 0.976947, 0 to 101.687590, 10 fc rgb "#0000FF"
set object 18 rect from 0.980963, 0 to 102.358053, 10 fc rgb "#FFFF00"
set object 19 rect from 0.987689, 0 to 103.328678, 10 fc rgb "#FF00FF"
set object 20 rect from 0.997032, 0 to 107.351870, 10 fc rgb "#808080"
set object 21 rect from 1.035679, 0 to 107.869814, 10 fc rgb "#800080"
set object 22 rect from 1.042665, 0 to 108.331107, 10 fc rgb "#008080"
set object 23 rect from 1.044967, 0 to 149.631390, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 1.471608, 10 to 158.278617, 20 fc rgb "#FF0000"
set object 25 rect from 1.262444, 10 to 132.495161, 20 fc rgb "#00FF00"
set object 26 rect from 1.279831, 10 to 133.007190, 20 fc rgb "#0000FF"
set object 27 rect from 1.282837, 10 to 133.603780, 20 fc rgb "#FFFF00"
set object 28 rect from 1.288603, 10 to 134.461831, 20 fc rgb "#FF00FF"
set object 29 rect from 1.296964, 10 to 138.232380, 20 fc rgb "#808080"
set object 30 rect from 1.333309, 10 to 138.690353, 20 fc rgb "#800080"
set object 31 rect from 1.339765, 10 to 139.139819, 20 fc rgb "#008080"
set object 32 rect from 1.341987, 10 to 152.246838, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 1.433570, 20 to 156.142931, 30 fc rgb "#FF0000"
set object 34 rect from 0.604437, 20 to 64.136172, 30 fc rgb "#00FF00"
set object 35 rect from 0.620803, 20 to 64.598399, 30 fc rgb "#0000FF"
set object 36 rect from 0.623454, 20 to 66.883493, 30 fc rgb "#FFFF00"
set object 37 rect from 0.645489, 20 to 68.049799, 30 fc rgb "#FF00FF"
set object 38 rect from 0.656714, 20 to 71.605161, 30 fc rgb "#808080"
set object 39 rect from 0.691190, 20 to 72.101628, 30 fc rgb "#800080"
set object 40 rect from 0.697709, 20 to 72.577032, 30 fc rgb "#008080"
set object 41 rect from 0.700417, 20 to 148.311838, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 1.404977, 30 to 175.305385, 40 fc rgb "#FF0000"
set object 43 rect from 0.185849, 30 to 22.255278, 40 fc rgb "#00FF00"
set object 44 rect from 0.218255, 30 to 50.294466, 40 fc rgb "#0000FF"
set object 45 rect from 0.484937, 30 to 51.005912, 40 fc rgb "#FFFF00"
set object 46 rect from 0.491815, 30 to 51.414187, 40 fc rgb "#FF00FF"
set object 47 rect from 0.495717, 30 to 52.032980, 40 fc rgb "#808080"
set object 48 rect from 0.501677, 30 to 52.138394, 40 fc rgb "#800080"
set object 49 rect from 0.503253, 30 to 52.293508, 40 fc rgb "#008080"
set object 50 rect from 0.504199, 30 to 145.059854, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 1.459273, 40 to 159.096723, 50 fc rgb "#FF0000"
set object 52 rect from 1.102754, 40 to 115.840146, 50 fc rgb "#00FF00"
set object 53 rect from 1.119073, 40 to 116.331943, 50 fc rgb "#0000FF"
set object 54 rect from 1.122098, 40 to 118.562669, 50 fc rgb "#FFFF00"
set object 55 rect from 1.143565, 40 to 119.798076, 50 fc rgb "#FF00FF"
set object 56 rect from 1.155663, 40 to 123.580038, 50 fc rgb "#808080"
set object 57 rect from 1.192120, 40 to 124.098604, 50 fc rgb "#800080"
set object 58 rect from 1.198997, 40 to 124.632837, 50 fc rgb "#008080"
set object 59 rect from 1.202139, 40 to 150.967958, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 1.420146, 50 to 148.789732, 60 fc rgb "#FF0000"
set object 61 rect from 0.533072, 50 to 55.596850, 60 fc rgb "#00FF00"
set object 62 rect from 0.536305, 50 to 55.731938, 60 fc rgb "#0000FF"
set object 63 rect from 0.537287, 50 to 56.135233, 60 fc rgb "#FFFF00"
set object 64 rect from 0.541237, 50 to 56.404684, 60 fc rgb "#FF00FF"
set object 65 rect from 0.543823, 50 to 56.981352, 60 fc rgb "#808080"
set object 66 rect from 0.549343, 50 to 57.064771, 60 fc rgb "#800080"
set object 67 rect from 0.550529, 50 to 57.153689, 60 fc rgb "#008080"
set object 68 rect from 0.551030, 50 to 146.835408, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
