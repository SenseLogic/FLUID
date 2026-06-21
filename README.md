![](https://github.com/senselogic/FLUID/blob/master/LOGO/fluid.png)

# Fluid

GPU-accelerated AI slow-motion video generator.

## Command line

```
fluid <input video file path> <output video file path> [<options>]
```

or

```
fluid_uv <input video file path> <output video file path> [<options>]
```

## Options

```
--factor <factor=4>
--compression <compression=22>
--scale <scale=1.0>
--mute
```

## Sample

```
fluid input_video.mp4 output_video.mp4
```

```
fluid input_video.mp4 output_video.mp4 --factor 2
```

```
fluid input_video.mp4 output_video.mp4 --compression 22
```

```
fluid input_video.mp4 output_video.mp4 --factor 8 --compression 22 --scale 0.5
```

```
fluid input_video.mp4 output_video.mp4 --mute
```

## Install

Run `install_ffmpeg.bat` then `install_packages.bat` or `install_uv_packages.bat`.

## Dependencies

- Python 3.10.11
- CUDA 12.4 (optional; CPU fallback)
- ffmpeg (in the path)
- RIFE 4.25 (`flownet.pkl`)

## Limitations

- Only processes `.mp4` input and output files.
- `--factor` defaults to 4 and must be an integer >= 2 (2, 4, 8, ... work best).

## Version

0.1

## Author

Eric Pelzer (ecstatic.coder@gmail.com).

## License

This project is licensed under the GNU General Public License version 3.

See the [LICENSE.md](LICENSE.md) file for details.
