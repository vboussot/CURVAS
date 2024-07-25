from pathlib import Path

from glob import glob
import time
import threading
import SimpleITK as sitk
import numpy as np
import torch
from tqdm import tqdm
from torch.cuda.amp import autocast

from model import BayesianUnetTotalSeg
from data import DataProcessing, image_to_data, get_patch_slices_from_shape, getData

INPUT_PATH = Path("/input")
OUTPUT_PATH = Path("/output")
RESOURCE_PATH = Path("resources")

def readImage(location: Path) -> sitk.Image:
    input_files = glob(str(location / "*.tiff")) + glob(str(location / "*.mha"))
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

    class_values = [7, 2, 3, 5] 
    nb_inference = 2

    data = torch.load(RESOURCE_PATH / "Data.pt")
    means, stds = data["means"], data["stds"]
    
    bayesian_model = BayesianUnetTotalSeg(channels=channels, nb_class=nb_class, means=means, stds=stds).to(0).eval() 
    bayesian_model.load_state_dict(torch.load(RESOURCE_PATH / "M291.pt"))

    data, attributes = image_to_data(input_image)
    data = torch.tensor(data).to(0)
    dataProcessing = DataProcessing()

    data = dataProcessing.pre_process(data, attributes)

    seg_patches = torch.zeros((4, *data.shape[1:]), dtype=torch.int16, device=0)
    div_patches = torch.zeros(data.shape[1:], dtype=torch.int16, device=0)

    patch_slices, _ = get_patch_slices_from_shape(patch_size, data.shape[1:], overlap)
    with autocast():
        with torch.no_grad():
            for _ in tqdm(range(nb_inference), desc='Processing'):
                bayesian_model.resample_bayesian_weights()
                for slices in patch_slices:
                    d = getData(data.unsqueeze(0), patch_size, slices)
                    _, layers = bayesian_model(d)
                    output = layers[-1][0]
                    seg_patches[[slice(None), *slices]] += torch.nn.functional.one_hot(torch.argmax(output[[slice(None), *[slice(0, slices[0].stop-slices[0].start), slice(0, slices[1].stop-slices[1].start), slice(0, slices[2].stop-slices[2].start)]]], dim=0), num_classes=nb_class).permute((3, 0, 1, 2)).index_select(0, torch.tensor(class_values, device=0))
                    div_patches[slices] += 1

    output = dataProcessing.post_process((torch.stack([seg_patches[0], seg_patches[1]+seg_patches[2], seg_patches[3]])/div_patches).cpu(), attributes).numpy()
    output_segmentation =  np.zeros(output.shape[1:], dtype=np.uint8)
    for i, prob in enumerate(output):
        output_segmentation[np.where(prob > 0.5)] = i+1
    return output_segmentation, *output

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
