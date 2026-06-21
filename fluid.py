#!/usr/bin/env python3

# -- IMPORTS

from __future__ import annotations;
from typing import Any;

try:

    import argparse;
    import ffmpeg;
    import mimetypes;
    import numpy as np;
    from math import exp;
    from pathlib import Path;
    import shutil;
    import subprocess;
    import sys;
    import torch;
    import torch.nn as nn;
    import torch.nn.functional as F;
    import warnings;
    from tqdm import tqdm;

except ImportError as import_error:

    print( f"Missing dependency: {import_error}", file=sys.stderr );
    print( "Install with:", file=sys.stderr );
    print( "  run install_packages.bat", file=sys.stderr );
    sys.exit( 1 );

warnings.filterwarnings( "ignore" );

# -- CONSTANTS

MODEL_NAME = "flownet.pkl";
MODEL_VERSION = 4.25;
DEFAULT_COMPRESSION = 22;
DEFAULT_FACTOR = 4;
DEFAULT_SCALE = 1.0;

APPLICATION_FOLDER_PATH = Path( __file__ ).resolve().parent;
MODEL_FOLDER_PATH = APPLICATION_FOLDER_PATH / "MODEL";

_backwarp_grid_by_cache_key_dictionary: dict[ str, torch.Tensor ] = {};
_create_window_3d_by_cache_key_dictionary: dict[ tuple, torch.Tensor ] = {};

# -- TYPES

class Head( nn.Module ):

    # -- CONSTRUCTORS

    def __init__(
        self
        ) -> None:

        super().__init__();
        self.cnn0 = nn.Conv2d( 3, 16, 3, 2, 1 );
        self.cnn1 = nn.Conv2d( 16, 16, 3, 1, 1 );
        self.cnn2 = nn.Conv2d( 16, 16, 3, 1, 1 );
        self.cnn3 = nn.ConvTranspose2d( 16, 4, 4, 2, 1 );
        self.relu = nn.LeakyReLU( 0.2, True );

    # -- OPERATIONS

    def forward(
        self,
        input_tensor: torch.Tensor,
        return_features: bool = False
        ) -> torch.Tensor | list[ torch.Tensor ]:

        first_encoder_layer_feature_tensor = self.cnn0( input_tensor );
        layer_output_tensor = self.relu( first_encoder_layer_feature_tensor );
        second_encoder_layer_feature_tensor = self.cnn1( layer_output_tensor );
        layer_output_tensor = self.relu( second_encoder_layer_feature_tensor );
        third_encoder_layer_feature_tensor = self.cnn2( layer_output_tensor );
        layer_output_tensor = self.relu( third_encoder_layer_feature_tensor );
        fourth_encoder_layer_feature_tensor = self.cnn3( layer_output_tensor );

        if return_features:

            return [
                first_encoder_layer_feature_tensor,
                second_encoder_layer_feature_tensor,
                third_encoder_layer_feature_tensor,
                fourth_encoder_layer_feature_tensor
                ];

        return fourth_encoder_layer_feature_tensor;

# ~~

class ResidualConvolutionBlock( nn.Module ):

    # -- CONSTRUCTORS

    def __init__(
        self,
        channel_count: int,
        dilation: int = 1
        ) -> None:

        super().__init__();
        self.conv = nn.Conv2d(
            channel_count,
            channel_count,
            3,
            1,
            dilation,
            dilation=dilation,
            groups=1
            );
        self.beta = nn.Parameter(
            torch.ones( ( 1, channel_count, 1, 1 ) ),
            requires_grad=True
            );
        self.relu = nn.LeakyReLU( 0.2, True );

    # -- OPERATIONS

    def forward(
        self,
        input_tensor: torch.Tensor
        ) -> torch.Tensor:

        return self.relu( self.conv( input_tensor ) * self.beta + input_tensor );

# ~~

class IntermediateFrameBlock( nn.Module ):

    # -- CONSTRUCTORS

    def __init__(
        self,
        input_channel_count: int,
        channel_count: int = 64,
        channel_count_override: int | None = None
        ) -> None:

        super().__init__();

        if channel_count_override is not None:

            channel_count = channel_count_override;

        self.conv0 = nn.Sequential(
            get_rife_convolution_layer( input_channel_count, channel_count // 2, 3, 2, 1 ),
            get_rife_convolution_layer( channel_count // 2, channel_count, 3, 2, 1 ),
            );
        self.convblock = nn.Sequential(
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            ResidualConvolutionBlock( channel_count ),
            );
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d( channel_count, 4 * 13, 4, 2, 1 ),
            nn.PixelShuffle( 2 )
            );

    # -- OPERATIONS

    def forward(
        self,
        input_tensor: torch.Tensor,
        flow_tensor: torch.Tensor | None = None,
        scale: float = 1
        ) -> tuple[ torch.Tensor, torch.Tensor, torch.Tensor ]:

        input_tensor = F.interpolate(
            input_tensor,
            scale_factor=1.0 / scale,
            mode="bilinear",
            align_corners=False
            );

        if flow_tensor is not None:

            flow_tensor = (
                F.interpolate(
                    flow_tensor,
                    scale_factor=1.0 / scale,
                    mode="bilinear",
                    align_corners=False
                    )
                * 1.0 / scale
                );
            input_tensor = torch.cat( ( input_tensor, flow_tensor ), 1 );

        feature_tensor = self.conv0( input_tensor );
        feature_tensor = self.convblock( feature_tensor );
        block_output_tensor = self.lastconv( feature_tensor );
        block_output_tensor = F.interpolate(
            block_output_tensor,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False
            );
        flow_tensor = block_output_tensor[ :, :4 ] * scale;
        mask_tensor = block_output_tensor[ :, 4:5 ];
        feature_tensor = block_output_tensor[ :, 5: ];

        return flow_tensor, mask_tensor, feature_tensor;

# ~~

class IntermediateFrameNetwork( nn.Module ):

    # -- CONSTRUCTORS

    def __init__(
        self
        ) -> None:

        super().__init__();
        self.block0 = IntermediateFrameBlock( 7 + 8, channel_count_override=192 );
        self.block1 = IntermediateFrameBlock( 8 + 4 + 8 + 8, channel_count_override=128 );
        self.block2 = IntermediateFrameBlock( 8 + 4 + 8 + 8, channel_count_override=96 );
        self.block3 = IntermediateFrameBlock( 8 + 4 + 8 + 8, channel_count_override=64 );
        self.block4 = IntermediateFrameBlock( 8 + 4 + 8 + 8, channel_count_override=32 );
        self.encode = Head();

    # -- OPERATIONS

    def forward(
        self,
        input_tensor: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
        scale_value_list: list[ float ] | None = None,
        training: bool = False,
        fast_mode: bool = True,
        ensemble: bool = False
        ) -> tuple[
            list[ torch.Tensor ],
            torch.Tensor,
            list[ torch.Tensor | tuple[ torch.Tensor, torch.Tensor ] ]
            ]:

        if scale_value_list is None:

            scale_value_list = [
                8,
                4,
                2,
                1
                ];

        channel_count = input_tensor.shape[ 1 ] // 2;
        first_source_image_tensor = input_tensor[ :, :channel_count ];
        second_source_image_tensor = input_tensor[ :, channel_count: ];

        if not torch.is_tensor( timestep ):

            timestep_tensor = (
                input_tensor[ :, :1 ].clone() * 0 + 1
                ) * timestep;

        else:

            timestep_tensor = timestep.repeat(
                1,
                1,
                first_source_image_tensor.shape[ 2 ],
                first_source_image_tensor.shape[ 3 ]
                );

        first_encoded_feature_tensor = self.encode( first_source_image_tensor[ :, :3 ] );
        second_encoded_feature_tensor = self.encode( second_source_image_tensor[ :, :3 ] );
        flow_tensor_list: list[ torch.Tensor ] = [];
        merged_frame_list: list[ torch.Tensor | tuple[ torch.Tensor, torch.Tensor ] ] = [];
        mask_tensor_list: list[ torch.Tensor ] = [];
        warped_first_source_image_tensor = first_source_image_tensor;
        warped_second_source_image_tensor = second_source_image_tensor;
        flow_tensor = None;
        mask_tensor = None;
        block_list = [
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4
            ];

        for block_index in range( 5 ):

            if flow_tensor is None:

                flow_tensor, mask_tensor, feature_tensor = block_list[ block_index ](
                    torch.cat(
                        (
                            first_source_image_tensor[ :, :3 ],
                            second_source_image_tensor[ :, :3 ],
                            first_encoded_feature_tensor,
                            second_encoded_feature_tensor,
                            timestep_tensor
                            ),
                        1
                        ),
                    None,
                    scale=scale_value_list[ block_index ]
                    );

                if ensemble:

                    print( "warning: ensemble is not supported since RIFEv4.21" );

            else:

                warped_first_encoded_feature_tensor = warp(
                    first_encoded_feature_tensor,
                    flow_tensor[ :, :2 ]
                    );
                warped_second_encoded_feature_tensor = warp(
                    second_encoded_feature_tensor,
                    flow_tensor[ :, 2:4 ]
                    );
                flow_delta_tensor, refined_mask_tensor, feature_tensor = block_list[ block_index ](
                    torch.cat(
                        (
                            warped_first_source_image_tensor[ :, :3 ],
                            warped_second_source_image_tensor[ :, :3 ],
                            warped_first_encoded_feature_tensor,
                            warped_second_encoded_feature_tensor,
                            timestep_tensor,
                            mask_tensor,
                            feature_tensor
                            ),
                        1
                        ),
                    flow_tensor,
                    scale=scale_value_list[ block_index ]
                    );

                if ensemble:

                    print( "warning: ensemble is not supported since RIFEv4.21" );

                else:

                    mask_tensor = refined_mask_tensor;

                flow_tensor = flow_tensor + flow_delta_tensor;

            mask_tensor_list.append( mask_tensor );
            flow_tensor_list.append( flow_tensor );
            warped_first_source_image_tensor = warp(
                first_source_image_tensor,
                flow_tensor[ :, :2 ]
                );
            warped_second_source_image_tensor = warp(
                second_source_image_tensor,
                flow_tensor[ :, 2:4 ]
                );
            merged_frame_list.append(
                (
                    warped_first_source_image_tensor,
                    warped_second_source_image_tensor
                    )
                );

        mask_tensor = torch.sigmoid( mask_tensor );
        merged_frame_list[ 4 ] = (
            warped_first_source_image_tensor * mask_tensor
            + warped_second_source_image_tensor * ( 1 - mask_tensor )
            );

        return flow_tensor_list, mask_tensor_list[ 4 ], merged_frame_list;

# ~~

class RifeModel:

    # -- CONSTRUCTORS

    def __init__(
        self
        ) -> None:

        self.flownet = IntermediateFrameNetwork();
        self.version = MODEL_VERSION;
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            );

    # -- OPERATIONS

    def eval(
        self
        ) -> None:

        self.flownet.eval();

    # ~~

    def to_device(
        self
        ) -> torch.device:

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            );
        self.flownet.to( self._device );

        return self._device;

    # ~~

    def load_weights(
        self,
        model_folder_path: Path | str
        ) -> None:

        model_weights_file_path = Path( model_folder_path ) / MODEL_NAME;

        if not model_weights_file_path.is_file():

            raise FileNotFoundError(
                f"RIFE weights not found: {model_weights_file_path}"
                );

        state_dict = torch.load(
            model_weights_file_path,
            map_location=( "cuda" if torch.cuda.is_available() else "cpu" )
            );
        state_dict_by_parameter_name_dictionary = {
            state_dict_key.replace( "module.", "" ): state_dict_value
            for state_dict_key, state_dict_value in state_dict.items()
            if "module." in state_dict_key
            };

        if state_dict_by_parameter_name_dictionary:

            self.flownet.load_state_dict( state_dict_by_parameter_name_dictionary, strict=False );

        else:

            self.flownet.load_state_dict( state_dict, strict=False );

    # ~~

    def inference(
        self,
        first_image_tensor: torch.Tensor,
        second_image_tensor: torch.Tensor,
        timestep: float = 0.5,
        scale: float = 1.0
        ) -> torch.Tensor:

        input_tensor = torch.cat( ( first_image_tensor, second_image_tensor ), 1 );
        scale_value_list = [
            16 / scale,
            8 / scale,
            4 / scale,
            2 / scale,
            1 / scale
            ];
        unused_flow_tensor, unused_mask_tensor, merged_frame_list = self.flownet(
            input_tensor,
            timestep,
            scale_value_list
            );

        return merged_frame_list[ -1 ];

# ~~

class VideoReader:

    # -- CONSTRUCTORS

    def __init__(
        self,
        arguments: argparse.Namespace
        ) -> None:

        self.arguments = arguments;
        input_type = mimetypes.guess_type( arguments.input_video_file_path )[ 0 ];

        if input_type is None or not input_type.startswith( "video" ):

            raise ValueError(
                f"Input must be a video file: {arguments.input_video_file_path}"
                );

        self.stream_reader = (
            ffmpeg.input( arguments.input_video_file_path )
                .output(
                    "pipe:",
                    format="rawvideo",
                    pix_fmt="bgr24",
                    loglevel="error"
                    )
                .run_async(
                    pipe_stdin=True,
                    pipe_stdout=True,
                    cmd=arguments.ffmpeg_binary_file_path
                    )
            );

        video_meta_info_by_field_dictionary = get_video_meta_info(
            arguments.input_video_file_path
            );
        self.width = video_meta_info_by_field_dictionary[ "width" ];
        self.height = video_meta_info_by_field_dictionary[ "height" ];
        self.input_frames_per_second = (
            video_meta_info_by_field_dictionary[ "frames_per_second" ]
            );
        self.display_aspect_ratio = (
            video_meta_info_by_field_dictionary[ "display_aspect_ratio" ]
            );
        self.sample_aspect_ratio = (
            video_meta_info_by_field_dictionary[ "sample_aspect_ratio" ]
            );
        self.has_audio = video_meta_info_by_field_dictionary[ "has_audio" ];
        self.frame_count = video_meta_info_by_field_dictionary[ "frame_count" ];

    # -- INQUIRIES

    def get_resolution(
        self
        ) -> tuple[ int, int ]:

        return self.height, self.width;

    # ~~

    def get_fps(
        self
        ) -> float:

        if self.arguments.frames_per_second_override is not None:

            return self.arguments.frames_per_second_override;

        return self.input_frames_per_second;

    # ~~

    def has_audio_stream(
        self
        ) -> bool:

        return self.has_audio;

    # ~~

    def get_display_aspect_ratio(
        self
        ) -> str | None:

        return self.display_aspect_ratio;

    # ~~

    def get_sample_aspect_ratio(
        self
        ) -> str | None:

        return self.sample_aspect_ratio;

    # ~~

    def __len__(
        self
        ) -> int:

        return self.frame_count;

    # -- OPERATIONS

    def get_frame(
        self
        ) -> np.ndarray | None:

        image_bytes = self.stream_reader.stdout.read(
            self.width * self.height * 3
            );

        if not image_bytes:

            return None;

        return (
            np.frombuffer( image_bytes, np.uint8 )
                .reshape(
                    [
                        self.height,
                        self.width,
                        3
                    ]
                )
            );

    # ~~

    def close(
        self
        ) -> None:

        self.stream_reader.stdin.close();
        self.stream_reader.wait();

# ~~

class VideoWriter:

    # -- CONSTRUCTORS

    def __init__(
        self,
        arguments: argparse.Namespace,
        output_height: int,
        output_width: int,
        output_video_file_path: str,
        frames_per_second: float,
        display_aspect_ratio: str | None = None,
        sample_aspect_ratio: str | None = None
        ) -> None:

        video_output_option_by_name_dictionary = {
            "pix_fmt": "yuv420p",
            "vcodec": "libx264",
            "crf": arguments.compression,
            "loglevel": "error"
            };

        video_output_option_by_name_dictionary.update(
            get_video_aspect_ratio_output_options(
                display_aspect_ratio,
                sample_aspect_ratio
                )
            );

        video_input = ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="bgr24",
            s=f"{output_width}x{output_height}",
            framerate=frames_per_second
            );

        self.stream_writer = (
            video_input.output(
                output_video_file_path,
                **video_output_option_by_name_dictionary
                )
            .overwrite_output()
            .run_async(
                pipe_stdin=True,
                pipe_stdout=True,
                cmd=arguments.ffmpeg_binary_file_path
                )
            );

    # -- OPERATIONS

    def write_frame(
        self,
        frame_array: np.ndarray
        ) -> None:

        self.stream_writer.stdin.write(
            frame_array.astype( np.uint8 ).tobytes()
            );

    # ~~

    def close( self ) -> None:

        self.stream_writer.stdin.close();
        self.stream_writer.wait();

# -- FUNCTIONS

def warp(
    input_tensor: torch.Tensor,
    flow_tensor: torch.Tensor
    ) -> torch.Tensor:

    device = flow_tensor.device;
    cache_key = ( str( device ), str( flow_tensor.size() ) );

    if cache_key not in _backwarp_grid_by_cache_key_dictionary:

        horizontal_coordinate_grid_tensor = (
            torch.linspace( -1.0, 1.0, flow_tensor.shape[ 3 ], device=device )
                .view( 1, 1, 1, flow_tensor.shape[ 3 ] )
                .expand( flow_tensor.shape[ 0 ], -1, flow_tensor.shape[ 2 ], -1 )
            );
        vertical_coordinate_grid_tensor = (
            torch.linspace( -1.0, 1.0, flow_tensor.shape[ 2 ], device=device )
                .view( 1, 1, flow_tensor.shape[ 2 ], 1 )
                .expand( flow_tensor.shape[ 0 ], -1, -1, flow_tensor.shape[ 3 ] )
            );
        _backwarp_grid_by_cache_key_dictionary[ cache_key ] = torch.cat(
            [
                horizontal_coordinate_grid_tensor,
                vertical_coordinate_grid_tensor
            ],
            1
            );

    flow_tensor = torch.cat(
        [
            flow_tensor[ :, 0:1, :, : ] / ( ( input_tensor.shape[ 3 ] - 1.0 ) / 2.0 ),
            flow_tensor[ :, 1:2, :, : ] / ( ( input_tensor.shape[ 2 ] - 1.0 ) / 2.0 ),
        ],
        1
        );
    sampling_grid = (
        _backwarp_grid_by_cache_key_dictionary[ cache_key ] + flow_tensor
        ).permute( 0, 2, 3, 1 );

    return F.grid_sample(
        input=input_tensor,
        grid=sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True
        );

# ~~

def get_gaussian_weight_tensor(
    window_size: int,
    sigma: float
    ) -> torch.Tensor:

    gaussian_weight_list = torch.Tensor(
        [
            exp(
                -( window_index - window_size // 2 ) ** 2
                / float( 2 * sigma ** 2 )
                )
            for window_index in range( window_size )
        ]
        );

    return gaussian_weight_list / gaussian_weight_list.sum();

# ~~

def get_three_dimensional_window_tensor(
    window_size: int,
    channel_count: int,
    device: torch.device
    ) -> torch.Tensor:

    one_dimensional_window_tensor = (
        get_gaussian_weight_tensor( window_size, 1.5 ).unsqueeze( 1 )
        );
    two_dimensional_window_tensor = (
        one_dimensional_window_tensor.mm( one_dimensional_window_tensor.t() )
        );
    three_dimensional_window_tensor = (
        two_dimensional_window_tensor.unsqueeze( 2 )
        @ one_dimensional_window_tensor.t()
        );
    window_tensor = (
        three_dimensional_window_tensor
            .expand( 1, channel_count, window_size, window_size, window_size )
            .contiguous()
            .to( device )
        );

    return window_tensor;

# ~~

def ssim_matlab(
    first_image_tensor: torch.Tensor,
    second_image_tensor: torch.Tensor,
    window_size: int = 11
    ) -> torch.Tensor:

    if torch.max( first_image_tensor ) > 128:

        value_range = 255;

    else:

        value_range = 1;

    _, _, height, width = first_image_tensor.size();
    effective_window_size = min( window_size, height, width );
    cache_key = ( effective_window_size, str( first_image_tensor.device ) );
    window_tensor = _create_window_3d_by_cache_key_dictionary.get( cache_key );

    if window_tensor is None:

        window_tensor = get_three_dimensional_window_tensor(
            effective_window_size,
            1,
            first_image_tensor.device
            );
        _create_window_3d_by_cache_key_dictionary[ cache_key ] = window_tensor;

    first_image_tensor = first_image_tensor.unsqueeze( 1 );
    second_image_tensor = second_image_tensor.unsqueeze( 1 );
    stability_constant_1 = ( 0.01 * value_range ) ** 2;
    stability_constant_2 = ( 0.03 * value_range ) ** 2;
    convolution_padding_tuple = ( 5, 5, 5, 5, 5, 5 );

    first_image_mean_tensor = F.conv3d(
        F.pad( first_image_tensor, convolution_padding_tuple, mode="replicate" ),
        window_tensor,
        padding=0,
        groups=1
        );
    second_image_mean_tensor = F.conv3d(
        F.pad( second_image_tensor, convolution_padding_tuple, mode="replicate" ),
        window_tensor,
        padding=0,
        groups=1
        );
    first_image_mean_squared_tensor = first_image_mean_tensor.pow( 2 );
    second_image_mean_squared_tensor = second_image_mean_tensor.pow( 2 );
    first_second_image_mean_product_tensor = (
        first_image_mean_tensor * second_image_mean_tensor
        );
    first_image_sigma_squared_tensor = (
        F.conv3d(
            F.pad(
                first_image_tensor * first_image_tensor,
                convolution_padding_tuple,
                mode="replicate"
                ),
            window_tensor,
            padding=0,
            groups=1
            )
        - first_image_mean_squared_tensor
        );
    second_image_sigma_squared_tensor = (
        F.conv3d(
            F.pad(
                second_image_tensor * second_image_tensor,
                convolution_padding_tuple,
                mode="replicate"
                ),
            window_tensor,
            padding=0,
            groups=1
            )
        - second_image_mean_squared_tensor
        );
    first_second_image_sigma_tensor = (
        F.conv3d(
            F.pad(
                first_image_tensor * second_image_tensor,
                convolution_padding_tuple,
                mode="replicate"
                ),
            window_tensor,
            padding=0,
            groups=1
            )
        - first_second_image_mean_product_tensor
        );
    numerator_term = 2.0 * first_second_image_sigma_tensor + stability_constant_2;
    denominator_term = (
        first_image_sigma_squared_tensor
        + second_image_sigma_squared_tensor
        + stability_constant_2
        );
    structural_similarity_tensor = (
        ( 2 * first_second_image_mean_product_tensor + stability_constant_1 )
        * numerator_term
        ) / (
        (
            first_image_mean_squared_tensor
            + second_image_mean_squared_tensor
            + stability_constant_1
            )
        * denominator_term
        );

    return structural_similarity_tensor.mean();

# ~~

def get_rife_convolution_layer(
    input_channel_count: int,
    output_channel_count: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1
    ) -> nn.Sequential:

    return nn.Sequential(
        nn.Conv2d(
            input_channel_count,
            output_channel_count,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True
            ),
        nn.LeakyReLU( 0.2, True )
        );

# ~~

def get_tensor_from_frame(
    frame_array: np.ndarray,
    device: torch.device
    ) -> torch.Tensor:

    return (
        torch.from_numpy( np.transpose( frame_array, ( 2, 0, 1 ) ) )
            .to( device, non_blocking=True )
            .unsqueeze( 0 )
            .float()
            / 255.0
        );

# ~~

def get_frame_from_tensor(
    frame_tensor: torch.Tensor,
    height: int,
    width: int
    ) -> np.ndarray:

    return (
        ( frame_tensor[ 0 ] * 255.0 )
            .byte()
            .cpu()
            .numpy()
            .transpose( 1, 2, 0 )[ :height, :width ]
        );

# ~~

def get_padding(
    height: int,
    width: int,
    scale: float
    ) -> tuple[ int, int, int, int ]:

    padding_divisor = max( 128, int( 128 / scale ) );
    padded_height = ( ( height - 1 ) // padding_divisor + 1 ) * padding_divisor;
    padded_width = ( ( width - 1 ) // padding_divisor + 1 ) * padding_divisor;

    return (
        0,
        padded_width - width,
        0,
        padded_height - height
        );

# ~~

def get_padded_frame_tensor(
    frame_tensor: torch.Tensor,
    frame_padding_tuple: tuple[ int, int, int, int ]
    ) -> torch.Tensor:

    return F.pad( frame_tensor, frame_padding_tuple );

# ~~

def get_intermediate_frame_tensor_list(
    model: RifeModel,
    first_frame_tensor: torch.Tensor,
    second_frame_tensor: torch.Tensor,
    intermediate_frame_count: int,
    scale: float
    ) -> list[ torch.Tensor ]:

    if model.version >= 3.9:

        return [
            model.inference(
                first_frame_tensor,
                second_frame_tensor,
                ( intermediate_frame_index + 1 ) * 1.0 / ( intermediate_frame_count + 1 ),
                scale
                )
            for intermediate_frame_index in range( intermediate_frame_count )
            ];

    middle_frame_tensor = model.inference(
        first_frame_tensor,
        second_frame_tensor,
        scale=scale
        );

    if intermediate_frame_count == 1:

        return [ middle_frame_tensor ];

    first_half_tensor_list = get_intermediate_frame_tensor_list(
        model,
        first_frame_tensor,
        middle_frame_tensor,
        intermediate_frame_count // 2,
        scale
        );
    second_half_tensor_list = get_intermediate_frame_tensor_list(
        model,
        middle_frame_tensor,
        second_frame_tensor,
        intermediate_frame_count // 2,
        scale
        );

    if intermediate_frame_count % 2:

        return [
            *first_half_tensor_list,
            middle_frame_tensor,
            *second_half_tensor_list
            ];

    return [
        *first_half_tensor_list,
        *second_half_tensor_list
        ];

# ~~

def get_rife_model(
    model_folder_path: Path | None = None
    ) -> RifeModel:

    resolved_model_folder_path = (
        model_folder_path
        if model_folder_path is not None
        else MODEL_FOLDER_PATH
        );
    model_weights_file_path = resolved_model_folder_path / MODEL_NAME;

    if not model_weights_file_path.is_file():

        print(
            f"RIFE weights not found: {model_weights_file_path}",
            file=sys.stderr
            );
        print(
            f"Place {MODEL_NAME} in {resolved_model_folder_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

    model = RifeModel();
    model.load_weights( resolved_model_folder_path );
    model.eval();
    model.to_device();

    return model;

# ~~

def get_ffprobe_command_path(
    ffmpeg_binary_file_path: str
    ) -> str:

    if ffmpeg_binary_file_path == "ffmpeg":

        return "ffprobe";

    binary_name = Path( ffmpeg_binary_file_path ).name;

    if "ffmpeg" in binary_name:

        return str(
            Path( ffmpeg_binary_file_path ).with_name(
                binary_name.replace( "ffmpeg", "ffprobe" )
                )
            );

    return "ffprobe";

# ~~

def is_valid_aspect_ratio(
    aspect_ratio: str | None
    ) -> bool:

    return (
        aspect_ratio is not None
        and aspect_ratio not in ( "N/A", "0:1" )
        );

# ~~

def get_video_aspect_ratio_output_options(
    display_aspect_ratio: str | None,
    sample_aspect_ratio: str | None
    ) -> dict[str, str]:

    if is_valid_aspect_ratio( display_aspect_ratio ):

        return { "aspect": display_aspect_ratio };

    if is_valid_aspect_ratio( sample_aspect_ratio ):

        return { "sar": sample_aspect_ratio };

    return {};

# ~~

def get_video_meta_info(
    input_video_file_path: str
    ) -> dict[str, Any]:

    probe_result = ffmpeg.probe( input_video_file_path );

    video_stream_list = [
        media_stream
        for media_stream in probe_result[ "streams" ]
        if media_stream[ "codec_type" ] == "video"
        ];

    has_audio = (
        any(
            media_stream[ "codec_type" ] == "audio"
            for media_stream in probe_result[ "streams" ]
            )
        );

    frame_count = video_stream_list[ 0 ].get( "nb_frames", "0" );

    if frame_count in ( "0", "N/A" ):

        frame_count = 0;

    video_stream = video_stream_list[ 0 ];

    return (
        {
            "width": video_stream[ "width" ],
            "height": video_stream[ "height" ],
            "frames_per_second": eval( video_stream[ "avg_frame_rate" ] ),
            "display_aspect_ratio": video_stream.get( "display_aspect_ratio" ),
            "sample_aspect_ratio": video_stream.get( "sample_aspect_ratio" ),
            "has_audio": has_audio,
            "frame_count": int( frame_count )
        }
        );

# ~~

def parse_arguments(
    ) -> argparse.Namespace:

    parser = (
        argparse.ArgumentParser(
            description="GPU-accelerated AI video slow-motion via RIFE frame interpolation",
            )
        );

    parser.add_argument(
        "input_video_file_path",
        type=Path,
        help="Input .mp4 file"
        );

    parser.add_argument(
        "output_video_file_path",
        type=Path,
        help="Output .mp4 file"
        );

    parser.add_argument(
        "--factor",
        type=int,
        default=DEFAULT_FACTOR,
        help=f"Slowdown factor (default: {DEFAULT_FACTOR}; 2, 4, 8, ...)"
        );

    parser.add_argument(
        "--compression",
        type=int,
        default=DEFAULT_COMPRESSION,
        help=f"Encoding compression (default: {DEFAULT_COMPRESSION}; higher = more compression)"
        );

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help=f"Flow estimation scale (default: {DEFAULT_SCALE}; use 0.5 for 1080p+/4K)"
        );

    parser.add_argument(
        "--mute",
        action="store_true",
        help="Omit audio from the output video"
        );

    return parser.parse_args();

# ~~

def validate_runtime(
    ) -> None:

    if shutil.which( "ffmpeg" ) is None:

        print(
            "ffmpeg not found. Install ffmpeg and add it to PATH.",
            file=sys.stderr
            );
        sys.exit( 1 );

    if shutil.which( get_ffprobe_command_path( "ffmpeg" ) ) is None:

        print(
            "ffprobe not found. Install ffmpeg with ffprobe.",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_input_video_file_path(
    input_video_file_path: Path
    ) -> None:

    if not input_video_file_path.is_file():

        print(
            f"Input video not found: {input_video_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

    if input_video_file_path.suffix.lower() != ".mp4":

        print(
            f"Input video must be an .mp4 file: {input_video_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_output_video_file_path(
    output_video_file_path: Path
    ) -> None:

    if output_video_file_path.suffix.lower() != ".mp4":

        print(
            f"Output video must be an .mp4 file: {output_video_file_path}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def validate_slowdown_factor(
    slowdown_factor: int
    ) -> None:

    if slowdown_factor < 2:

        print(
            f"Slowdown factor must be an integer >= 2: {slowdown_factor}",
            file=sys.stderr
            );
        sys.exit( 1 );

# ~~

def get_atempo_filter(
    slowdown_factor: int
    ) -> str:

    target_tempo = 1.0 / slowdown_factor;
    atempo_filter_list = [];
    remaining_tempo = target_tempo;

    while abs( remaining_tempo - 1.0 ) > 0.001:

        if remaining_tempo < 0.5:

            atempo_filter_list.append( "atempo=0.5" );
            remaining_tempo /= 0.5;

        elif remaining_tempo > 2.0:

            atempo_filter_list.append( "atempo=2.0" );
            remaining_tempo /= 2.0;

        else:

            atempo_filter_list.append( f"atempo={remaining_tempo:.6f}" );
            remaining_tempo = 1.0;

    return ",".join( atempo_filter_list );

# ~~

def merge_audio(
    input_video_file_path: Path,
    video_only_file_path: Path,
    output_video_file_path: Path,
    slowdown_factor: int
    ) -> None:

    atempo_filter = get_atempo_filter( slowdown_factor );
    temporary_output_video_file_path = (
        output_video_file_path.with_suffix( ".tmp.mp4" )
        );

    ffmpeg_command_list = [
        "ffmpeg",
        "-y",
        "-i", str( video_only_file_path ),
        "-i", str( input_video_file_path ),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-filter:a", atempo_filter,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str( temporary_output_video_file_path )
        ];

    subprocess_result = subprocess.run(
        ffmpeg_command_list,
        capture_output=True,
        text=True
        );

    if subprocess_result.returncode != 0:

        print( "Audio merge failed; output will have no audio.", file=sys.stderr );
        shutil.move( str( video_only_file_path ), str( output_video_file_path ) );

        return;

    video_only_file_path.unlink();
    shutil.move(
        str( temporary_output_video_file_path ),
        str( output_video_file_path )
        );

# ~~

def slowdown_mp4(
    input_video_file_path: Path,
    output_video_file_path: Path,
    model: RifeModel,
    slowdown_factor: int,
    compression: int,
    scale: float,
    mute: bool = False
    ) -> None:

    output_video_file_path.parent.mkdir( parents=True, exist_ok=True );

    video_reader_argument_namespace = (
        argparse.Namespace(
            input_video_file_path=str( input_video_file_path.resolve() ),
            ffmpeg_binary_file_path="ffmpeg",
            frames_per_second_override=None
            )
        );

    video_writer_argument_namespace = (
        argparse.Namespace(
            ffmpeg_binary_file_path="ffmpeg",
            compression=compression
            )
        );

    print( f"Reading {input_video_file_path}" );

    video_reader = VideoReader( video_reader_argument_namespace );
    input_height, input_width = video_reader.get_resolution();
    input_frames_per_second = video_reader.get_fps();
    has_audio = video_reader.has_audio_stream() and not mute;

    if scale not in ( 0.25, 0.5, 1.0, 2.0, 4.0 ):

        print(
            f"Warning: unusual flow estimation scale {scale}; recommended values are 0.25, 0.5, 1.0, 2.0, 4.0",
            file=sys.stderr
            );

    frame_padding_tuple = get_padding( input_height, input_width, scale );
    device = model.to_device();

    video_only_file_path = (
        output_video_file_path.with_suffix( ".noaudio.mp4" )
        if has_audio
        else output_video_file_path
        );

    video_writer = None;
    progress_bar = None;

    print( f"Writing {output_video_file_path}" );

    try:

        video_writer = (
            VideoWriter(
                video_writer_argument_namespace,
                input_height,
                input_width,
                str( video_only_file_path ),
                input_frames_per_second,
                video_reader.get_display_aspect_ratio(),
                video_reader.get_sample_aspect_ratio()
                )
            );

        first_frame_array = video_reader.get_frame();

        if first_frame_array is None:

            print( "Input video contains no frames.", file=sys.stderr );
            sys.exit( 1 );

        last_frame_array = first_frame_array;
        last_frame_tensor = get_padded_frame_tensor(
            get_tensor_from_frame( last_frame_array, device ),
            frame_padding_tuple
            );

        progress_bar = (
            tqdm(
                total=len( video_reader ) or None,
                unit="frame",
                desc="fluid"
                )
            );

        video_writer.write_frame( last_frame_array );
        progress_bar.update( 1 );

        intermediate_frame_count = slowdown_factor - 1;

        while True:

            next_frame_array = video_reader.get_frame();

            if next_frame_array is None:

                break;

            first_frame_tensor = last_frame_tensor;
            second_frame_tensor = get_padded_frame_tensor(
                get_tensor_from_frame( next_frame_array, device ),
                frame_padding_tuple
                );

            first_frame_tensor_small = (
                F.interpolate(
                    first_frame_tensor,
                    ( 32, 32 ),
                    mode="bilinear",
                    align_corners=False
                    )
                );
            second_frame_tensor_small = (
                F.interpolate(
                    second_frame_tensor,
                    ( 32, 32 ),
                    mode="bilinear",
                    align_corners=False
                    )
                );
            structural_similarity_value = ssim_matlab(
                first_frame_tensor_small[ :, :3 ],
                second_frame_tensor_small[ :, :3 ]
                );

            should_break_frame_loop = False;
            working_frame_array = next_frame_array;
            working_frame_tensor = second_frame_tensor;

            if structural_similarity_value > 0.996:

                skipped_frame_array = video_reader.get_frame();

                if skipped_frame_array is None:

                    should_break_frame_loop = True;
                    working_frame_array = last_frame_array;
                    working_frame_tensor = first_frame_tensor;

                else:

                    working_frame_array = skipped_frame_array;
                    working_frame_tensor = get_padded_frame_tensor(
                        get_tensor_from_frame( working_frame_array, device ),
                        frame_padding_tuple
                        );
                    working_frame_tensor = model.inference(
                        first_frame_tensor,
                        working_frame_tensor,
                        scale=scale
                        );
                    working_frame_tensor_small = (
                        F.interpolate(
                            working_frame_tensor,
                            ( 32, 32 ),
                            mode="bilinear",
                            align_corners=False
                            )
                        );
                    structural_similarity_value = ssim_matlab(
                        first_frame_tensor_small[ :, :3 ],
                        working_frame_tensor_small[ :, :3 ]
                        );
                    working_frame_array = get_frame_from_tensor(
                        working_frame_tensor,
                        input_height,
                        input_width
                        );

            if structural_similarity_value < 0.2:

                intermediate_output_tensor_list = (
                    [ first_frame_tensor ] * intermediate_frame_count
                    );

            else:

                try:

                    intermediate_output_tensor_list = get_intermediate_frame_tensor_list(
                        model,
                        first_frame_tensor,
                        working_frame_tensor,
                        intermediate_frame_count,
                        scale
                        );

                except RuntimeError as runtime_error:

                    print( f"Error: {runtime_error}", file=sys.stderr );
                    print(
                        "Try again with --scale 0.5.",
                        file=sys.stderr
                        );

                    if output_video_file_path.is_file():

                        output_video_file_path.unlink();

                    sys.exit( 1 );

            for intermediate_output_tensor in intermediate_output_tensor_list:

                video_writer.write_frame(
                    get_frame_from_tensor(
                        intermediate_output_tensor,
                        input_height,
                        input_width
                        )
                    );

            video_writer.write_frame( working_frame_array );

            if device.type == "cuda":

                torch.cuda.synchronize( device );

            last_frame_array = working_frame_array;
            last_frame_tensor = working_frame_tensor;
            progress_bar.update( 1 );

            if should_break_frame_loop:

                break;

    finally:

        if progress_bar is not None:

            progress_bar.close();

        if video_writer is not None:

            video_writer.close();

        video_reader.close();

    if has_audio:

        merge_audio(
            input_video_file_path,
            video_only_file_path,
            output_video_file_path,
            slowdown_factor
            );

# ~~

def main(
    ) -> None:

    arguments = parse_arguments();
    validate_runtime();
    validate_input_video_file_path( arguments.input_video_file_path );
    validate_output_video_file_path( arguments.output_video_file_path );
    validate_slowdown_factor( arguments.factor );

    if not torch.cuda.is_available():

        print( "CUDA not available; using CPU (slow).", file=sys.stderr );

    torch.set_grad_enabled( False );

    if torch.cuda.is_available():

        torch.backends.cudnn.enabled = True;
        torch.backends.cudnn.benchmark = True;

    model = get_rife_model( MODEL_FOLDER_PATH );

    slowdown_mp4(
        arguments.input_video_file_path,
        arguments.output_video_file_path,
        model,
        slowdown_factor=arguments.factor,
        compression=arguments.compression,
        scale=arguments.scale,
        mute=arguments.mute
        );

# -- STATEMENTS

if __name__ == "__main__":

    main();
