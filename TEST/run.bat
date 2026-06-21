if not exist "OUT\race_2x.mp4" call "..\fluid.bat" IN\race.mp4 OUT\race_2x.mp4 --factor 2 --compression 22
if not exist "OUT\race_2x_muted.mp4" call "..\fluid.bat" IN\race.mp4 OUT\race_2x_muted.mp4 --factor 2 --compression 22 --mute
if not exist "OUT\race_4x.mp4" call "..\fluid.bat" IN\race.mp4 OUT\race_4x.mp4 --factor 4 --compression 22
if not exist "OUT\race_8x.mp4" call "..\fluid.bat" IN\race.mp4 OUT\race_8x.mp4 --factor 4 --compression 22
