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

set object 15 rect from 0.148527, 0 to 155.469728, 10 fc rgb "#FF0000"
set object 16 rect from 0.097141, 0 to 101.319114, 10 fc rgb "#00FF00"
set object 17 rect from 0.099607, 0 to 102.056331, 10 fc rgb "#0000FF"
set object 18 rect from 0.100063, 0 to 102.875232, 10 fc rgb "#FFFF00"
set object 19 rect from 0.100880, 0 to 104.035173, 10 fc rgb "#FF00FF"
set object 20 rect from 0.102008, 0 to 105.341126, 10 fc rgb "#808080"
set object 21 rect from 0.103296, 0 to 105.902715, 10 fc rgb "#800080"
set object 22 rect from 0.104093, 0 to 106.461242, 10 fc rgb "#008080"
set object 23 rect from 0.104381, 0 to 150.972914, 10 fc rgb "#000080"
set arrow from  0,0 to 158.400000,0 nohead
set object 24 rect from 0.143187, 10 to 152.710781, 20 fc rgb "#FF0000"
set object 25 rect from 0.040462, 10 to 44.695464, 20 fc rgb "#00FF00"
set object 26 rect from 0.044300, 10 to 46.088209, 20 fc rgb "#0000FF"
set object 27 rect from 0.045258, 10 to 47.934310, 20 fc rgb "#FFFF00"
set object 28 rect from 0.047106, 10 to 49.492470, 20 fc rgb "#FF00FF"
set object 29 rect from 0.048605, 10 to 51.239527, 20 fc rgb "#808080"
set object 30 rect from 0.050336, 10 to 52.171769, 20 fc rgb "#800080"
set object 31 rect from 0.051705, 10 to 53.020278, 20 fc rgb "#008080"
set object 32 rect from 0.052073, 10 to 145.322291, 20 fc rgb "#000080"
set arrow from  0,10 to 158.400000,10 nohead
set object 33 rect from 0.146750, 20 to 153.810482, 30 fc rgb "#FF0000"
set object 34 rect from 0.080793, 20 to 84.263099, 30 fc rgb "#00FF00"
set object 35 rect from 0.082884, 20 to 85.034007, 30 fc rgb "#0000FF"
set object 36 rect from 0.083398, 20 to 85.872311, 30 fc rgb "#FFFF00"
set object 37 rect from 0.084225, 20 to 87.152737, 30 fc rgb "#FF00FF"
set object 38 rect from 0.085462, 20 to 88.360667, 30 fc rgb "#808080"
set object 39 rect from 0.086666, 20 to 88.969229, 30 fc rgb "#800080"
set object 40 rect from 0.087526, 20 to 89.673768, 30 fc rgb "#008080"
set object 41 rect from 0.087941, 20 to 149.256489, 30 fc rgb "#000080"
set arrow from  0,20 to 158.400000,20 nohead
set object 42 rect from 0.145156, 30 to 152.952783, 40 fc rgb "#FF0000"
set object 43 rect from 0.063393, 30 to 66.933432, 40 fc rgb "#00FF00"
set object 44 rect from 0.065980, 30 to 67.745185, 40 fc rgb "#0000FF"
set object 45 rect from 0.066467, 30 to 68.821397, 40 fc rgb "#FFFF00"
set object 46 rect from 0.067547, 30 to 70.535780, 40 fc rgb "#FF00FF"
set object 47 rect from 0.069227, 30 to 71.784555, 40 fc rgb "#808080"
set object 48 rect from 0.070449, 30 to 72.548319, 40 fc rgb "#800080"
set object 49 rect from 0.071434, 30 to 73.531612, 40 fc rgb "#008080"
set object 50 rect from 0.072160, 30 to 147.419579, 40 fc rgb "#000080"
set arrow from  0,30 to 158.400000,30 nohead
set object 51 rect from 0.151725, 40 to 158.705508, 50 fc rgb "#FF0000"
set object 52 rect from 0.128574, 40 to 133.034689, 50 fc rgb "#00FF00"
set object 53 rect from 0.130645, 40 to 133.689197, 50 fc rgb "#0000FF"
set object 54 rect from 0.131043, 40 to 134.573447, 50 fc rgb "#FFFF00"
set object 55 rect from 0.131909, 40 to 135.736450, 50 fc rgb "#FF00FF"
set object 56 rect from 0.133050, 40 to 136.891286, 50 fc rgb "#808080"
set object 57 rect from 0.134205, 40 to 137.540690, 50 fc rgb "#800080"
set object 58 rect from 0.135068, 40 to 138.127806, 50 fc rgb "#008080"
set object 59 rect from 0.135396, 40 to 154.441501, 50 fc rgb "#000080"
set arrow from  0,40 to 158.400000,40 nohead
set object 60 rect from 0.150210, 50 to 157.239251, 60 fc rgb "#FF0000"
set object 61 rect from 0.112788, 50 to 116.922146, 60 fc rgb "#00FF00"
set object 62 rect from 0.114853, 50 to 117.634856, 60 fc rgb "#0000FF"
set object 63 rect from 0.115321, 50 to 118.530334, 60 fc rgb "#FFFF00"
set object 64 rect from 0.116213, 50 to 119.704570, 60 fc rgb "#FF00FF"
set object 65 rect from 0.117365, 50 to 120.908418, 60 fc rgb "#808080"
set object 66 rect from 0.118548, 50 to 121.541484, 60 fc rgb "#800080"
set object 67 rect from 0.119420, 50 to 122.195992, 60 fc rgb "#008080"
set object 68 rect from 0.119785, 50 to 152.714867, 60 fc rgb "#000080"
set arrow from  0,50 to 158.400000,50 nohead
plot 0
