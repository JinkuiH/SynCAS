import numpy as np
from dataloading.dataset import SimpleITKIO
from typing import Tuple, Union,List, Callable
from scipy.ndimage import map_coordinates
from skimage.transform import resize
from copy import deepcopy
from sampling import resample_data_or_seg_to_shape
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import List, Iterator, Dict
import os
from predictor import export_prediction_from_logits, convert_predicted_logits_to_segmentation_with_correct_shape
from multiprocessing import Pool
from time import sleep
import multiprocessing
from dataloading.utils import compute_gaussian, compute_steps_for_sliding_window, empty_cache, dummy_context
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from acvl_utils.cropping_and_padding.padding import pad_nd_image
# Hello! crop_to_nonzero is the function you are looking for. Ignore the rest.
from acvl_utils.cropping_and_padding.bounding_boxes import get_bbox_from_mask, bounding_box_to_slice
from scipy.ndimage import binary_fill_holes
from tqdm import tqdm
import itertools
from predictor import Predictor,compute_metrics_on_folder
from datetime import datetime
from time import time, sleep
import csv
import math
import os
import re
from pathlib import Path
# from models_pytorch.model.model_ours import S2CAC
from tqdm import tqdm
from time import sleep
import gc
from os.path import join
#import psutil  # 用于内存监控（可选）

default_num_processes = 2


def create_nonzero_mask(data):
    """

    :param data:
    :return: the mask is True where the data is nonzero
    """
    assert data.ndim in (3, 4), "data must have shape (C, X, Y, Z) or shape (C, X, Y)"
    nonzero_mask = data[0] != 0
    for c in range(1, data.shape[0]):
        nonzero_mask |= data[c] != 0
    return binary_fill_holes(nonzero_mask)


def crop_to_nonzero(data, seg=None, nonzero_label=-1):
    """

    :param data:
    :param seg:
    :param nonzero_label: this will be written into the segmentation map
    :return:
    """
    nonzero_mask = create_nonzero_mask(data)
    bbox = get_bbox_from_mask(nonzero_mask)
    slicer = bounding_box_to_slice(bbox)
    nonzero_mask = nonzero_mask[slicer][None]
    
    slicer = (slice(None), ) + slicer
    data = data[slicer]
    if seg is not None:
        seg = seg[slicer]
        seg[(seg == 0) & (~nonzero_mask)] = nonzero_label
    else:
        seg = np.where(nonzero_mask, np.int8(0), np.int8(nonzero_label))
    return data, seg, bbox



def check_workers_alive_and_busy(export_pool: Pool, worker_list: List, results_list: List, allowed_num_queued: int = 0):
    """

    returns True if the number of results that are not ready is greater than the number of available workers + allowed_num_queued
    """
    alive = [i.is_alive() for i in worker_list]
    if not all(alive):
        raise RuntimeError('Some background workers are no longer alive')

    not_ready = [not i.ready() for i in results_list]
    if sum(not_ready) >= (len(export_pool._pool) + allowed_num_queued):
        return True
    return False


    
def compute_new_shape(old_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                      old_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                      new_spacing: Union[Tuple[float, ...], List[float], np.ndarray]) -> np.ndarray:
    assert len(old_spacing) == len(old_shape)
    assert len(old_shape) == len(new_spacing)
    new_shape = np.array([int(round(i / j * k)) for i, j, k in zip(old_spacing, new_spacing, old_shape)])
    return new_shape



class CTNormalization():
    def __init__(self, plans):
        self.plans = plans

    def run(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        window_level = 50
        window_width = 400
        lower_bound = window_level - window_width / 2  # -150
        upper_bound = window_level + window_width / 2  # 250

        image = image.astype(np.float32, copy=False)
        np.clip(image, lower_bound, upper_bound, out=image)

        # Optionally normalize to [-1, 1] or [0, 1]; here we use mean-std normalization:
        mean_intensity = (upper_bound + lower_bound) / 2  # should be 50
        std_intensity = window_width / 2  # should be 200
        image -= mean_intensity
        image /= max(std_intensity, 1e-8)
        return image

class InferencePreprocessor(object):
    def __init__(self, plans, verbose: bool = False,device: torch.device = torch.device('cuda'),
                 tile_step_size: float = 0.5,
                 use_gaussian: bool = True,
                 use_mirroring: bool = True,
                 ):
        self.verbose = verbose
        self.device = device
        self.plans = plans
        self._normalize = CTNormalization(plans)
        self.network = None
        self.perform_everything_on_device = True

        self.tile_step_size = tile_step_size
        self.use_gaussian = use_gaussian
        self.use_mirroring = use_mirroring
        self.allow_tqdm=False
        self.allowed_mirroring_axes = None
        self.output_folder = None

    def load_checkpoint(self, filename_or_checkpoint: Union[dict, str]) -> None:
        # if not self.was_initialized:
        #     self.initialize()

        if isinstance(filename_or_checkpoint, str):
            checkpoint = torch.load(filename_or_checkpoint, map_location=self.device, weights_only=False)
        # if state dict comes from nn.DataParallel but we use non-parallel model here then the state dict keys do not
        # match. Use heuristic to make it match
        new_state_dict = {}
        for k, value in checkpoint['network_weights'].items():
            key = k
            if key not in self.network.state_dict().keys() and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value

        self.my_init_kwargs = checkpoint['init_args']
        self.current_epoch = checkpoint['current_epoch']
        # self.logger.load_checkpoint(checkpoint['logging'])
        self._best_ema = checkpoint['_best_ema']
        self.inference_allowed_mirroring_axes = checkpoint[
            'inference_allowed_mirroring_axes'] if 'inference_allowed_mirroring_axes' in checkpoint.keys() else self.inference_allowed_mirroring_axes

        
        self.network.load_state_dict(new_state_dict)

        # self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        # if self.grad_scaler is not None:
        #     if checkpoint['grad_scaler_state'] is not None:
        #         self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state'])

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict, plans_manager):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        # apply transpose_forward, this also needs to be applied to the spacing!
        data = data.transpose([0, *[i + 1 for i in plans_manager["transpose_forward"]]])
        if seg is not None:
            seg = seg.transpose([0, *[i + 1 for i in plans_manager["transpose_forward"]]])
        original_spacing = [properties['spacing'][i] for i in plans_manager["transpose_forward"]]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        # this command will generate a segmentation. This is important because of the nonzero mask which we may need
        data, seg, bbox = crop_to_nonzero(data, seg)
        properties['bbox_used_for_cropping'] = bbox
        # print(data.shape, seg.shape)
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample
        target_spacing = plans_manager['spacing']  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        # normalize
        # normalization MUST happen before resampling or we get huge problems with resampled nonzero masks no
        # longer fitting the images perfectly!
        data = self._normalize.run(data, seg)

        # print('current shape', data.shape[1:], 'current_spacing', original_spacing,
        #       '\ntarget shape', new_shape, 'target_spacing', target_spacing)
        old_shape = data.shape[1:]
        data = resample_data_or_seg_to_shape(data,
                                            new_shape,
                                            original_spacing,
                                            target_spacing)
        
        # resample_data_or_seg(data, new_shape, is_seg=False, axis=3, order=3, order_z=0, do_separate_z=True)
        seg = None

      
        return data, seg

    def run_case(self, image_files, seg_file: Union[str, None], plans_manager):
        """
        seg file can be none (test cases)

        order of operations is: transpose -> crop -> resample
        so when we export we need to run the following order: resample -> crop -> transpose (we could also run
        transpose at a different place, but reverting the order of operations done during preprocessing seems cleaner)
        """
        # print(f"Running case for image files {image_files}")
        rw = SimpleITKIO()

        # load image(s)
        data, data_properties = rw.read_images([image_files])

        # if possible, load seg
        if seg_file is not None:
            # print(f"Loading seg file {seg_file}")
            seg, _ = rw.read_seg(seg_file)
        else:
            # print(f"No seg file provided")
            seg = None
        # print(f"Running case for {image_files}")
        data, seg = self.run_case_npy(data, seg, data_properties, plans_manager)
        # print(f"Done running case for {image_files}")
        data = torch.from_numpy(data).to(dtype=torch.float32, memory_format=torch.contiguous_format)
        # print(f"Done converting to torch for {image_files}")
        return data, seg, data_properties
    
    def predict_NCCT_from_nested_folders(self, root_folder: str, save_probabilities: bool = False):
        """
        遍历 root_folder 下的所有子文件夹，自动预测以 NCCT.nii.gz 结尾的图像文件，
        并将结果保存至对应子文件夹中，输出文件名前加 CA_。

        Args:
            root_folder (str): 包含待处理子文件夹的根目录。
            save_probabilities (bool): 是否保存概率图（可选）。
        """
        image_file_lists = []
        output_paths = []

        # 遍历子目录，查找以 NCCT.nii.gz 结尾的文件
        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith("NCCT.nii.gz"):
                    input_path = os.path.join(subdir, file)
                    output_filename = f"CA_{file}"
                    output_path = os.path.join(subdir, output_filename)

                    image_file_lists.append(input_path)  # 需要是 list of list[str]
                    output_paths.append(output_path)

        if not image_file_lists:
            print("No NCCT.nii.gz files found.")
            return

        self.output_folder = ''  # 让 export_prediction_from_logits 使用传入的完整路径保存
        data_iterator = self.get_image_iter(image_file_lists, self.plans)

        # 覆盖每个 data_iterator 中的 'ofile' 字段为实际输出路径
        for i in range(len(data_iterator)):
            data_iterator[i]['ofile'] = output_paths[i]

        return self.predict_from_data_iterator(data_iterator, save_probabilities)


    def predict_from_data_iterator(self,
                                   data_iterator,
                                   save_probabilities: bool = False,
                                   num_processes_segmentation_export: int = 1):
        """
        each element returned by data_iterator must be a dict with 'data', 'ofile' and 'data_properties' keys!
        If 'ofile' is None, the result will be returned instead of written to a file
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            r = []
            for preprocessed in tqdm(data_iterator, desc="Running model predictions", unit="case"):
                data = preprocessed['data']
                if isinstance(data, str):
                    delfile = data
                    data = torch.from_numpy(np.load(data))
                    os.remove(delfile)

                ofile = preprocessed['ofile']
                # if ofile is not None:
                #     print(f'\nPredicting {os.path.basename(ofile)}:')
                # else:
                #     print(f'\nPredicting image of shape {data.shape}:')
                properties = preprocessed['data_properties']

                # let's not get into a runaway situation where the GPU predicts so fast that the disk has to b swamped with
                # npy files
                proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(export_pool, worker_list, r, allowed_num_queued=2)

                prediction = self.predict_logits_from_preprocessed_data(data).cpu()


                # this needs to go into background processes
                # export_prediction_from_logits(prediction, properties, self.configuration_manager, self.plans_manager,
                #                               self.dataset_json, ofile, save_probabilities)
                # print('sending off prediction to background worker for resampling and export')
                r.append(
                    export_pool.starmap_async(
                        export_prediction_from_logits,
                        ((prediction, properties, self.plans, join(self.output_folder,ofile), save_probabilities),)
                    )
                )
               
                # if ofile is not None:
                #     print(f'done with {os.path.basename(ofile)}')
                # else:
                #     print(f'\nDone with image of shape {data.shape}:')
            # ret = [i.get()[0] for i in r]
            ret = []
            for future in tqdm(r, desc="Exporting predictions", unit="case"):
                ret.append(future.get()[0])


        # clear lru cache
        compute_gaussian.cache_clear()
        # clear device cache
        empty_cache(self.device)
        return ret
    
    def get_image_iter_2(self, image_files_list, plans_manager):
        """
        Generator: yields processed data for each case to avoid excessive memory usage.
        """
        for image_files in image_files_list:
            data, _, data_properties = self.run_case(image_files, None, plans_manager)
            basename = os.path.basename(image_files if isinstance(image_files, str) else image_files[0])
            yield {
                "data": data,
                "data_properties": data_properties,
                "ofile": basename
            }


    def predict_NCCT_from_nested_folders_2(self, root_folder: str, save_probabilities: bool = False):
        """
        Predict CA segmentation for NCCT files under root_folder, saving results per case without excessive memory use.
        """
        image_file_lists = []
        output_paths = []

        for subdir, _, files in os.walk(root_folder):
            for file in files:
                if file.endswith("NCCT.nii.gz") and not file.startswith("CA_"):
                    input_path = os.path.join(subdir, file)
                    output_filename = f"CA_{file}"
                    output_path = os.path.join(subdir, output_filename)

                    image_file_lists.append(input_path)
                    output_paths.append(output_path)

        if not image_file_lists:
            print("No NCCT.nii.gz files found.")
            return

        self.output_folder = ''
        data_iterator = self.get_image_iter_2(image_file_lists, self.plans)

        return self.predict_from_data_iterator_2(data_iterator, output_paths, save_probabilities)


    def predict_from_data_iterator_2(self, data_iterator, output_paths, save_probabilities=False, num_processes_segmentation_export=1):
        """
        Process each case one-by-one using a generator-based data_iterator.
        """
        with multiprocessing.get_context("spawn").Pool(num_processes_segmentation_export) as export_pool:
            worker_list = [i for i in export_pool._pool]
            results = []

            for i, preprocessed in enumerate(tqdm(data_iterator, desc="Running model predictions", unit="case")):
                data = preprocessed['data']
                properties = preprocessed['data_properties']
                
                # 模型推理
                prediction = self.predict_logits_from_preprocessed_data(data).cpu()

                if len(output_paths) > 1:
                    ofile = output_paths[i]
                    # 异步导出
                    results.append(
                        export_pool.starmap_async(
                            export_prediction_from_logits,
                            ((prediction, properties, self.plans, ofile, save_probabilities),)
                        )
                    )
                else:
                    ofile = preprocessed['ofile']
                    results.append(
                    export_pool.starmap_async(
                        export_prediction_from_logits,
                        ((prediction, properties, self.plans, join(self.output_folder,ofile), save_probabilities),)
                    )
                )

                

                # 控制 export 队列长度
                while check_workers_alive_and_busy(export_pool, worker_list, results, allowed_num_queued=1):
                    sleep(0.1)

                # 强制释放内存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # 可选: 打印内存使用
                # process = psutil.Process(os.getpid())
                # print(f"Memory usage: {process.memory_info().rss / (1024 ** 3):.2f} GB")

            # 等待所有导出完成
            final_results = []
            for future in tqdm(results, desc="Exporting predictions", unit="case"):
                final_results.append(future.get()[0])

        # 清理缓存
        compute_gaussian.cache_clear()
        empty_cache(self.device)

        return final_results

    def get_image_iter(self, image_files_list, plans_manager):
        """
        Returns all processed image data as a list without using DataLoader.
        
        Args:
            image_files_list: A list where each item is a list of image file paths for one case.
            plans_manager: The plans manager object required for processing.
            
        Returns:
            List[Dict[str, torch.Tensor]]: A list of dictionaries containing the image data and associated properties.
        """
        all_data = []  # Create an empty list to store all the processed data
        
        for image_files in image_files_list:
            # Process each set of image files using the processor
            data, _, data_properties = self.run_case(image_files, None, plans_manager)
            basename = os.path.basename(image_files)
            ofile = [f.replace(".nii.gz", "_processed.nii.gz").replace("nii", "_processed.nii.gz") for f in image_files]  # Example output filename
            # print(f"Processed {basename}")
            
            # Append the result as a dictionary to the all_data list
            all_data.append({
                "data": data,
                "data_properties": data_properties,
                "ofile": basename
            })
        
        return all_data
   
    def get_all_files_in_directory(self,directory_path: str):
        """
        Returns a list of all file paths in the given directory and its subdirectories.
        
        Args:
            directory_path (str): The path of the directory to search.
            
        Returns:
            List[str]: A list of complete file paths.
        """
        file_paths = []
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_paths.append(os.path.join(root, file))  # Get the full file path
                # print(f"Found {file}")
        return file_paths
    
    def predict_from_files(self,
                           source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None,
                           num_parts: int = 1,
                           part_id: int = 0):
        self.output_folder = output_folder_or_list_of_truncated_output_files
        os.makedirs(self.output_folder, exist_ok=True)
        
        list_of_lists_or_source_folder = self.get_all_files_in_directory(source_folder)
        # print(f"Found {len(list_of_lists_or_source_folder)} files")
        if len(list_of_lists_or_source_folder) == 0:
            return

        data_iterator = self.get_image_iter_2(list_of_lists_or_source_folder, self.plans)

        return self.predict_from_data_iterator_2(data_iterator, output_paths=[self.output_folder])
    
    def _internal_get_sliding_window_slicers(self, image_size: Tuple[int, ...]):
        slicers = []
        if len(self.plans['patch_size']) < len(image_size):
            assert len(self.plans['patch_size']) == len(
                image_size) - 1, 'if tile_size has less entries than image_size, ' \
                                 'len(tile_size) ' \
                                 'must be one shorter than len(image_size) ' \
                                 '(only dimension ' \
                                 'discrepancy of 1 allowed).'
            steps = compute_steps_for_sliding_window(image_size[1:], self.plans['patch_size'],
                                                     self.tile_step_size)
            if self.verbose: print(f'n_steps {image_size[0] * len(steps[0]) * len(steps[1])}, image size is'
                                   f' {image_size}, tile_size {1}, '
                                   f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for d in range(image_size[0]):
                for sx in steps[0]:
                    for sy in steps[1]:
                        slicers.append(
                            tuple([slice(None), d, *[slice(si, si + ti) for si, ti in
                                                     zip((sx, sy), self.plans['patch_size'])]]))
        else:
            steps = compute_steps_for_sliding_window(image_size, self.plans['patch_size'],
                                                     self.tile_step_size)
            if self.verbose: print(
                f'n_steps {np.prod([len(i) for i in steps])}, image size is {image_size}, tile_size {1}, '
                f'tile_step_size {self.tile_step_size}\nsteps:\n{steps}')
            for sx in steps[0]:
                for sy in steps[1]:
                    for sz in steps[2]:
                        slicers.append(
                            tuple([slice(None), *[slice(si, si + ti) for si, ti in
                                                  zip((sx, sy, sz), self.plans['patch_size'])]]))
        return slicers
    
    def predict_sliding_window_return_logits(self, input_image: torch.Tensor) \
            -> Union[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            assert isinstance(input_image, torch.Tensor)
            self.network = self.network.to(self.device)
            self.network.eval()

            empty_cache(self.device)

            # Autocast can be annoying
            # If the device_type is 'cpu' then it's slow as heck on some CPUs (no auto bfloat16 support detection)
            # and needs to be disabled.
            # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False
            # is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
            # So autocast will only be active if we have a cuda device.
            with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
                assert input_image.ndim == 4, 'input_image must be a 4D np.ndarray or torch.Tensor (c, x, y, z)'

                if self.verbose:
                    print(f'Input shape: {input_image.shape}')

                # if input_image is smaller than tile_size we need to pad it to tile_size.
                data, slicer_revert_padding = pad_nd_image(input_image, self.plans['patch_size'],
                                                           'constant', {'value': 0}, True,
                                                           None)

                slicers = self._internal_get_sliding_window_slicers(data.shape[1:])

                if self.perform_everything_on_device and self.device != 'cpu':
                    # we need to try except here because we can run OOM in which case we need to fall back to CPU as a results device
                    try:
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                               self.perform_everything_on_device)
                    except RuntimeError:
                        print(
                            'Prediction on device was unsuccessful, probably due to a lack of memory. Moving results arrays to CPU')
                        empty_cache(self.device)
                        predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers, False)
                else:
                    predicted_logits = self._internal_predict_sliding_window_return_logits(data, slicers,
                                                                                           self.perform_everything_on_device)

                empty_cache(self.device)
                # revert padding
                predicted_logits = predicted_logits[(slice(None), *slicer_revert_padding[1:])]
        return predicted_logits
    
    def _internal_predict_sliding_window_return_logits(self,
                                                       data: torch.Tensor,
                                                       slicers,
                                                       do_on_device: bool = True,
                                                       ):
        predicted_logits = n_predictions = prediction = gaussian = workon = None
        results_device = self.device if do_on_device else torch.device('cpu')

        try:
            empty_cache(self.device)

            # move data to device
            if self.verbose:
                print(f'move image to device {results_device}')
            data = data.to(results_device)

            # preallocate arrays
            if self.verbose:
                print(f'preallocating results arrays on device {results_device}')
            num_segmentation_heads = len(self.plans['annotated_classes_key'])
            predicted_logits = torch.zeros((num_segmentation_heads, *data.shape[1:]),
                                           dtype=torch.half,
                                           device=results_device)
            n_predictions = torch.zeros(data.shape[1:], dtype=torch.half, device=results_device)

            if self.use_gaussian:
                gaussian = compute_gaussian(tuple(self.plans['patch_size']), sigma_scale=1. / 8,
                                            value_scaling_factor=10,
                                            device=results_device)
            else:
                gaussian = 1

            if not self.allow_tqdm and self.verbose:
                print(f'running prediction: {len(slicers)} steps')
            for sl in tqdm(slicers, disable=not self.allow_tqdm):
                workon = data[sl][None]
                workon = workon.to(self.device)

                prediction = self._internal_maybe_mirror_and_predict(workon)[0].to(results_device)

                if self.use_gaussian:
                    prediction *= gaussian
                # print(predicted_logits.shape, prediction.shape)

                predicted_logits[sl] += prediction.squeeze()
                n_predictions[sl[1:]] += gaussian

            predicted_logits /= n_predictions
            # check for infs
            if torch.any(torch.isinf(predicted_logits)):
                raise RuntimeError('Encountered inf in predicted array. Aborting... If this problem persists, '
                                   'reduce value_scaling_factor in compute_gaussian or increase the dtype of '
                                   'predicted_logits to fp32')
        except Exception as e:
            del predicted_logits, n_predictions, prediction, gaussian, workon
            empty_cache(self.device)
            empty_cache(results_device)
            raise e
        return predicted_logits
    
    def _internal_maybe_mirror_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        mirror_axes = self.allowed_mirroring_axes if self.use_mirroring else None
        prediction = self.network(x)

        if mirror_axes is not None:
            # check for invalid numbers in mirror_axes
            # x should be 5d for 3d images and 4d for 2d. so the max value of mirror_axes cannot exceed len(x.shape) - 3
            assert max(mirror_axes) <= x.ndim - 3, 'mirror_axes does not match the dimension of the input!'

            mirror_axes = [m + 2 for m in mirror_axes]
            axes_combinations = [
                c for i in range(len(mirror_axes)) for c in itertools.combinations(mirror_axes, i + 1)
            ]
            for axes in axes_combinations:
                prediction += torch.flip(self.network(torch.flip(x, axes)), axes)
            prediction /= (len(axes_combinations) + 1)
        return prediction
     
    def predict_logits_from_preprocessed_data(self, data: torch.Tensor) -> torch.Tensor:
        """
        IMPORTANT! IF YOU ARE RUNNING THE CASCADE, THE SEGMENTATION FROM THE PREVIOUS STAGE MUST ALREADY BE STACKED ON
        TOP OF THE IMAGE AS ONE-HOT REPRESENTATION! SEE PreprocessAdapter ON HOW THIS SHOULD BE DONE!

        RETURNED LOGITS HAVE THE SHAPE OF THE INPUT. THEY MUST BE CONVERTED BACK TO THE ORIGINAL IMAGE SIZE.
        SEE convert_predicted_logits_to_segmentation_with_correct_shape
        """
        n_threads = torch.get_num_threads()
        torch.set_num_threads(default_num_processes if default_num_processes < n_threads else n_threads)
        prediction = None


        prediction = self.predict_sliding_window_return_logits(data).to('cpu')



        if self.verbose: print('Prediction done')
        torch.set_num_threads(n_threads)
        return prediction
    def print_to_log_file(self, *args, also_print_to_console=True, add_timestamp=True):
        
        timestamp = time()
        dt_object = datetime.fromtimestamp(timestamp)

        if add_timestamp:
            args = (f"{dt_object}:", *args)

        successful = False
        max_attempts = 5
        ctr = 0
        while not successful and ctr < max_attempts:
            try:
                with open(self.log_file, 'a+') as f:
                    for a in args:
                        f.write(str(a))
                        f.write(" ")
                    f.write("\n")
                successful = True
            except IOError:
                print(f"{datetime.fromtimestamp(timestamp)}: failed to log: ", sys.exc_info())
                sleep(0.5)
                ctr += 1
        if also_print_to_console:
            print(*args)

    
    def calculate_averages(self, metrics):

        metric_keys = ['Dice', 'IoU', 'HD95', 'MSD', 'Agatston_Ref', 'Agatston_Pred',
                    'FP', 'TP', 'FN', 'TN', 'n_pred', 'n_ref','clDice', 'clPrecision', 'clRecall']

        # 初始化每个指标的值列表
        values = {k: [] for k in metric_keys}

        # 用于 Macro 计算的单样本指标列表
        precision_list = []
        recall_list = []
        specificity_list = []
        f1_list = []

        eps = 1e-8  # 防止除以零

        for item in metrics['metric_per_case']:
            m = item['metrics'][1]  # 假设你关心类别1

            for k in metric_keys:
                # print(m.get('Dice', 0))
                val = m.get(k, 0)

                # 特殊处理 nan 的情况
                if k in ['Dice', 'IoU']:
                    val = 1.0 if val != val else val  # nan -> 1.0
                elif k in ['HD95', 'MSD']:
                    val = 0.0 if val != val else val  # nan -> 0.0

                values[k].append(val)

            # 获取单样本 TP, FP, FN, TN
            TP = m.get('TP', 0)
            FP = m.get('FP', 0)
            FN = m.get('FN', 0)
            TN = m.get('TN', 0)

            # 每个样本的指标
            precision = TP / (TP + FP + eps)
            recall = TP / (TP + FN + eps)
            specificity = TN / (TN + FP + eps)
            f1 = 2 * precision * recall / (precision + recall + eps)

            precision_list.append(precision)
            recall_list.append(recall)
            specificity_list.append(specificity)
            f1_list.append(f1)

        # 计算均值和标准差
        averages = {k: float(np.mean(values[k])) for k in metric_keys}
        stds = {k: float(np.std(values[k], ddof=1)) for k in metric_keys}  # 样本标准差

        # 添加 Macro-average 指标
        averages['Precision'] = float(np.mean(precision_list))
        averages['Recall'] = float(np.mean(recall_list))
        averages['Specificity'] = float(np.mean(specificity_list))
        averages['F1'] = float(np.mean(f1_list))

        # 添加 Macro-average 指标 (标准差)
        stds['Precision'] = float(np.std(precision_list, ddof=1))
        stds['Recall'] = float(np.std(recall_list, ddof=1))
        stds['Specificity'] = float(np.std(specificity_list, ddof=1))
        stds['F1'] = float(np.std(f1_list, ddof=1))

        

        return averages, stds
    
    def calculate_Matrics_from_folder(self, folders):
        
        orininal_data_folder = folders['original_data_folder']
        GT_folder = folders['gt_segmentations_folder']
        validation_output_folder = folders['validation_output_folder']

        timestamp = datetime.now()
        self.log_file = join(validation_output_folder, "testing_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))

        default_num_processes = 8
        
        metrics = compute_metrics_on_folder(orininal_data_folder, GT_folder,
                                            validation_output_folder,
                                            join(validation_output_folder, 'summary.json'),
                                            '.nii.gz',
                                            self.plans['foreground_labels'],
                                            num_processes=default_num_processes)
        self.print_to_log_file("Validation complete", also_print_to_console=True)
 
        averages,stds = self.calculate_averages(metrics)
        self.print_to_log_file("Mean Validation Recall: {:.3f} ± {:.3f}".format(averages['Recall'], stds['Recall']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation Precision: {:.3f} ± {:.3f}".format(averages['Precision'], stds['Precision']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation Dice: {:.3f} ± {:.3f}".format(averages['Dice'], stds['Dice']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation MSD: {:.3f} ± {:.3f}".format(averages['MSD'], stds['MSD']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation HD95: {:.3f} ± {:.3f}".format(averages['HD95'], stds['HD95']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation clRecall: {:.3f} ± {:.3f}".format(averages['clRecall'], stds['clRecall']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation clPrecision: {:.3f} ± {:.3f}".format(averages['clPrecision'], stds['clPrecision']), also_print_to_console=True)
        self.print_to_log_file("Mean Validation clDice: {:.3f} ± {:.3f}".format(averages['clDice'], stds['clDice']), also_print_to_console=True)


        return  averages['clDice']


if __name__ =='__main__':
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    from models.network import Unet


    plans = load_json('config.json')

    Inferencer = InferencePreprocessor(plans)
    

    
    print('Inferencer loaded')
    Inferencer.network = Unet(dimension=3, input_nc=1, output_nc=len(Inferencer.plans['annotated_classes_key']), num_downs=4, ngf=16).to(Inferencer.device)
    print('Network loaded')
    path = 'weights/checkpoint_SynCAS_w_finetune.pth'
    Inferencer.load_checkpoint(path)
    print('Model loaded')

    
    source_folder = 'data/Images'
    target_folder = 'outputs/results2'
    # GT_folder = 'data/GT'

    Inferencer.predict_from_files(source_folder, target_folder)

    # Inferencer.calculate_Matrics_from_folder({
    #     'original_data_folder': source_folder,
    #     'gt_segmentations_folder': GT_folder,
    #     'validation_output_folder': target_folder
    # })



