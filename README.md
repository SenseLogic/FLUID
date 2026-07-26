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
--slowdown-factor <slowdown_factor=4>
--flow-estimation-scale <flow_estimation_scale=1.0>
--crop <left_distance> <right_distance> <top_distance> <bottom_distance>
--max-width <maximum_width=0>
--max-height <maximum_height=0>
--compression <compression=22>
--mute
--skip
--cpu
--cuda
--rocm
```

If none of `--cpu`, `--cuda`, or `--rocm` is passed, Fluid uses CUDA when available, otherwise ROCm when available, otherwise CPU.

## Sample

```
fluid input_video.mp4 output_video.mp4
```

```
fluid input_video.mp4 output_video.mp4 --slowdown-factor 2
```

```
fluid input_video.mp4 output_video.mp4 --compression 22
```

```
fluid input_video.mp4 output_video.mp4 --slowdown-factor 8 --compression 22 --flow-estimation-scale 0.5
```

```
fluid input_video.mp4 output_video.mp4 --mute
```

## Install

Run `install_ffmpeg.bat`, then one of:

- `install_packages_cuda.bat` / `install_uv_packages_cuda.bat` — NVIDIA CUDA
- `install_packages_rocm.bat` / `install_uv_packages_rocm.bat` — AMD ROCm
- `install_packages_cpu.bat` / `install_uv_packages_cpu.bat` — CPU only

## Dependencies

- Python 3.12.10
- CUDA 12.4 (NVIDIA) or ROCm 7.2.1 (AMD), optional
- ffmpeg (in the path)
- RIFE 4.25 (`flownet.pkl`)

## Limitations

- Only processes `.mp4` input and output files.
- `--slowdown-factor` defaults to 4 and must be an integer >= 2 (2, 4, 8, ... work best).

## Version

0.1

## Author

Eric Pelzer (ecstatic.coder@gmail.com).

## License

This project is licensed under the GNU General Public License version 3.

See the [LICENSE.md](LICENSE.md) file for details.
