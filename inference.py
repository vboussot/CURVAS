from pathlib import Path

from glob import glob
import time
import threading
import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast
import copy

from model import BayesianUnetTotalSeg
from data import DataProcessing, image_to_data, get_patch_slices_from_shape, getData, Accumulator, Cosinus

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def readImage(location: Path) -> sitk.Image:
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha")) + glob(str(location / "*.nii.gz"))
    return sitk.ReadImage(input_files[0])

def run():

    input_image = readImage(INPUT_PATH / "images/thoracic-abdominal-ct")
    
    _show_torch_cuda_info()
    
    start_time = time.time()
        
    output_abdominal_organ_segmentation, output_pancreas_confidence, output_kidney_confidence, output_liver_confidence = perform_inference(input_image)
    
    prediction_time = np.round((time.time() - start_time)/60, 2)
    print('Prediction time: '+str(prediction_time))
    
    print('Saving the predictions')
    
    start_time = time.time()
    
    write_files_in_parallel([(Path(OUTPUT_PATH / "images/abdominal-organ-segmentation"), output_abdominal_organ_segmentation),
                             (Path(OUTPUT_PATH / "images/kidney-confidence"), output_kidney_confidence),
                             (Path(OUTPUT_PATH / "images/pancreas-confidence"), output_pancreas_confidence),
                             (Path(OUTPUT_PATH / "images/liver-confidence"), output_liver_confidence)
                             ], input_image)
    
    saving_time = np.round((time.time() - start_time)/60, 2)
    print('Saving time: '+str(saving_time))
    
    print('Finished running algorithm!')
    
    return 0


def perform_inference(input_image: sitk.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:   
    nb_class = 25
    channels = [1, 32, 64, 128, 256, 320, 320]
    patch_size = [128,128,128]
    overlap = 20

    class_values = {7:1, 2:2, 3:2, 5:3}
    nb_inference = 2

    data = torch.load(RESOURCE_PATH / "Data.pt")
    means, stds = data["means"], data["stds"]
    
    bayesian_model = BayesianUnetTotalSeg(channels=channels, nb_class=nb_class, means=means, stds=stds).to(0).eval() 
    bayesian_model.load_state_dict(torch.load(RESOURCE_PATH / "M291.pt"))

    data, attributes = image_to_data(input_image)
    data = torch.tensor(data).to(0)
    dataProcessing = DataProcessing()

    data = dataProcessing.pre_process(data, attributes)
    patch_slices, _ = get_patch_slices_from_shape(patch_size, data.shape[1:], overlap)

    accumulator = Accumulator(patch_slices, patch_size, overlap, nb_class)

    with autocast():
        with torch.no_grad():
            for i in range(nb_inference):
                bayesian_model.resample_bayesian_weights()
                for index, slices in enumerate(patch_slices):
                    d = getData(data.unsqueeze(0), patch_size, slices)
                    _, layers = bayesian_model(d)
                    output = layers[-1][0].cpu()
                    accumulator.addLayer(index, output)
                print("Progress : {} | {}".format(i, nb_inference))
    seg_patches = accumulator.getResult()/nb_inference
    output = torch.argmax(seg_patches, dim=0)
    output_segmentation = torch.zeros(data.shape[1:], dtype=torch.uint8)
    for old_class, new_class in class_values.items():
        output_segmentation[output == old_class] = new_class
    output_segmentation = dataProcessing.post_process(output_segmentation.unsqueeze(0), copy.deepcopy(attributes)).squeeze(0).numpy()

    return output_segmentation, dataProcessing.post_process(seg_patches[7].unsqueeze(0), copy.deepcopy(attributes)).squeeze(0).numpy(), dataProcessing.post_process((seg_patches[2]+seg_patches[3]).unsqueeze(0), copy.deepcopy(attributes)).squeeze(0).numpy(), dataProcessing.post_process(seg_patches[5].unsqueeze(0), copy.deepcopy(attributes)).squeeze(0).numpy()

def write_array_as_image_file(location: Path, data: np.ndarray, initial_image: sitk.Image):
    location.mkdir(parents=True, exist_ok=True)
    suffix = ".mha"
    image = sitk.GetImageFromArray(data)
    image.CopyInformation(initial_image)
    sitk.WriteImage(
        image,
        location / f"output{suffix}",
        useCompression=True,
    )
    

def write_files_in_parallel(files_data: list[tuple[Path, np.ndarray]], initial_image: sitk.Image):
    threads = []
    for location, data in files_data:
        print('location: '+str(location))
        thread = threading.Thread(target=write_array_as_image_file, kwargs={"location" : location, "data" : data, "initial_image" : initial_image})
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


def _show_torch_cuda_info():
    import torch

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
