import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
from typing import Any, Union
import copy
import itertools
from functools import partial

class Attribute(dict[str, Any]):

    def __init__(self, attributes : dict[str, Any] = {}) -> None:
        super().__init__()
        for k, v in attributes.items():
            super().__setitem__(copy.deepcopy(k), copy.deepcopy(v))
    
    def __getitem__(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return str(super().__getitem__("{}_{}".format(key, i-1)))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def __setitem__(self, key: str, value: Any) -> None:
        if "_" not in key:
            i = len([k for k in super().keys() if k.startswith(key)])
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace('\n', '')
            super().__setitem__("{}_{}".format(key, i), result)
        else:
            result = None
            if isinstance(value, torch.Tensor):
                result = str(value.numpy())
            else:
                result = str(value)
            result = result.replace('\n', '')
            super().__setitem__(key, result)

    def pop(self, key: str) -> Any:
        i = len([k for k in super().keys() if k.startswith(key)])
        if i > 0 and "{}_{}".format(key, i-1) in super().keys():   
            return super().pop("{}_{}".format(key, i-1))
        else:
            raise NameError("{} not in cache_attribute".format(key))

    def get_np_array(self, key) -> np.ndarray:
        return np.fromstring(self[key][1:-1], sep=" ", dtype=np.double)
    
    def get_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.get_np_array(key))
    
    def pop_np_array(self, key):
        return np.fromstring(self.pop(key)[1:-1], sep=" ", dtype=np.double)
    
    def pop_tensor(self, key) -> torch.Tensor:
        return torch.tensor(self.pop_np_array(key))
    
    def __contains__(self, key: str) -> bool:
        return len([k for k in super().keys() if k.startswith(key)]) > 0
    
    def isInfo(self, key: str, value: str) -> bool:
        return key in self and self[key] == value

def _resample_affine(data: torch.Tensor, matrix: torch.Tensor):
    mode = "nearest" if data.dtype == torch.uint8 else "bilinear"
    return F.grid_sample(data.unsqueeze(0).type(torch.float32), F.affine_grid(matrix[:, :-1,...].type(torch.float32).to(data.device), [1]+list(data.shape), align_corners=True), align_corners=True, mode=mode, padding_mode="reflection").squeeze(0).type(data.dtype)

def _affine_matrix(matrix: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return torch.cat((torch.cat((matrix, translation.unsqueeze(0).T), dim=1), torch.tensor([[0, 0, 0, 1]])), dim=0)

class Canonical():

    def __init__(self) -> None:
        self.canonical_direction = torch.diag(torch.tensor([-1, -1, 1])).to(torch.double)

    def __call__(self, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        matrix = _affine_matrix(self.canonical_direction @ cache_attribute.get_tensor("Direction").reshape(3,3).inverse(), torch.tensor([0, 0, 0]))
        cache_attribute["Direction"] = (self.canonical_direction).flatten()
        center = np.asarray([(input.shape[-i-1]-1) * cache_attribute.get_np_array("Spacing")[i] for i in range(3)])/2+cache_attribute.get_np_array("Origin")
        translation_center = self.canonical_direction @ (self.canonical_direction.inverse() @ (center)-center)
        cache_attribute["Origin"] =  self.canonical_direction @ cache_attribute.get_np_array("Origin")+translation_center
        return _resample_affine(input, matrix.unsqueeze(0))
    
    def inverse(self, input: torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        cache_attribute.pop("Direction")
        cache_attribute.pop("Origin")
        matrix = _affine_matrix((self.canonical_direction @ cache_attribute.get_tensor("Direction").reshape(3,3).inverse()).inverse(), torch.tensor([0, 0, 0]))
        return _resample_affine(input, matrix.unsqueeze(0))

class ResampleIsotropic():

    def __init__(self, spacing : list[float] = [1., 1., 1.]) -> None:
        self.spacing = torch.tensor(spacing, dtype=torch.float64)
        
    def _resample(self, input: torch.Tensor, size: list[int]) -> torch.Tensor:
        args = {}
        if input.dtype == torch.uint8:
            mode = "nearest"
        elif len(input.shape) < 4:
            mode = "bilinear"
        else:
            mode = "trilinear"
        return F.interpolate(input.type(torch.float32).unsqueeze(0), size=tuple(size), mode=mode).squeeze(0).type(input.dtype)
    
    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        resize_factor = self.spacing/cache_attribute.get_tensor("Spacing").flip(0)
        cache_attribute["Spacing"] = self.spacing.flip(0)
        cache_attribute["Size"] = np.asarray([int(x) for x in torch.tensor(input.shape[1:])])
        size = [int(x) for x in (torch.tensor(input.shape[1:]) * 1/resize_factor)]
        cache_attribute["Size"] = np.asarray(size)
        return self._resample(input, size)
    
    def inverse(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        size_0 = cache_attribute.pop_np_array("Size")
        size_1 = cache_attribute.pop_np_array("Size")
        _ = cache_attribute.pop_np_array("Spacing")
        return self._resample(input, [int(size) for size in size_1])
    
def clip_and_standardize(data: torch.Tensor) -> torch.Tensor:
    data[data < -1024] = -1024
    data[data > 276.0] = 276
    return  (data-(-370.00039267657144))/436.5998675471528


def image_to_data(image: sitk.Image) -> tuple[np.ndarray, Attribute]:
    attributes = Attribute()
    attributes["Origin"] = np.asarray(image.GetOrigin())
    attributes["Spacing"] = np.asarray(image.GetSpacing())
    attributes["Direction"] = np.asarray(image.GetDirection())
    data = sitk.GetArrayFromImage(image)

    if image.GetNumberOfComponentsPerPixel() == 1:
        data = np.expand_dims(data, 0)
    else:
        data = np.transpose(data, (len(data.shape)-1, *[i for i in range(len(data.shape)-1)]))
    return data, attributes

class Padding():

    def __init__(self, padding : list[int] = [0,0,0,0,0,0], mode : str = "default:constant,reflect,replicate,circular") -> None:
        self.padding = padding
        self.mode = mode

    def __call__(self, input : torch.Tensor, cache_attribute: Attribute) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            origin = torch.tensor(cache_attribute.get_np_array("Origin"))
            matrix = torch.tensor(cache_attribute.get_np_array("Direction").reshape((len(origin),len(origin))))
            origin = torch.matmul(origin, matrix)
            for dim in range(len(self.padding)//2):
                origin[-dim-1] -= self.padding[dim*2]* cache_attribute.get_np_array("Spacing")[-dim-1]
            cache_attribute["Origin"] = torch.matmul(origin, torch.inverse(matrix))
        result = F.pad(input.unsqueeze(0), tuple(self.padding), self.mode.split(":")[0], float(self.mode.split(":")[1]) if len(self.mode.split(":")) == 2 else 0).squeeze(0)
        return result

    def inverse(self, input : torch.Tensor, cache_attribute: dict[str, torch.Tensor]) -> torch.Tensor:
        if "Origin" in cache_attribute and "Spacing" in cache_attribute and "Direction" in cache_attribute:
            cache_attribute.pop("Origin")
        slices = [slice(0, shape) for shape in input.shape]
        for dim in range(len(self.padding)//2):
            slices[-dim-1] = slice(self.padding[dim*2], input.shape[-dim-1]-self.padding[dim*2+1])
        result = input[slices]
        return result
    

class DataProcessing():

    def __init__(self) -> None:
        self.canonical = Canonical()
        self.resampleIsotropic = ResampleIsotropic([1.5, 1.5, 1.5])
        self.padding = Padding([20]*6, "constant:-1.5")

    def pre_process(self, data: torch.Tensor, attributes: Attribute) -> torch.Tensor:
        data = self.canonical(data, attributes)
        data = self.resampleIsotropic(data, attributes)
        data = clip_and_standardize(data)
        data = self.padding(data, attributes)
        return data
    
    def post_process(self, data: torch.Tensor, attributes: Attribute) -> torch.Tensor:
        data = self.padding.inverse(data, attributes)
        data = self.resampleIsotropic.inverse(data, attributes)
        data = self.canonical.inverse(data, attributes)
        return data
    
def get_patch_slices_from_shape(patch_size: list[int], shape : list[int], overlap: Union[int, None]) -> tuple[list[tuple[slice]], list[tuple[int, bool]]]:
    if len(shape) != len(patch_size):
        return [tuple([slice(0, s) for s in shape])], [(1, True)]*len(shape)
    
    patch_slices = []
    nb_patch_per_dim = []
    slices : list[list[slice]] = []
    if overlap is None:
        size = [np.ceil(a/b) for a, b in zip(shape, patch_size)]
        tmp = np.zeros(len(size), dtype=np.int_)
        for i, s in enumerate(size):
            if s > 1:
                tmp[i] = np.mod(patch_size[i]-np.mod(shape[i], patch_size[i]), patch_size[i])//(size[i]-1)
        overlap = tmp
    else:
        overlap = [overlap if size > 1 else 0 for size in patch_size]
    
    for dim in range(len(shape)):
        assert overlap[dim] < patch_size[dim],  "Overlap must be less than patch size"

    for dim in range(len(shape)):
        slices.append([])
        index = 0
        while True:
            start = (patch_size[dim]-overlap[dim])*index

            end = start + patch_size[dim]
            if end >= shape[dim]:
                end = shape[dim]
                slices[dim].append(slice(start, end))
                break
            slices[dim].append(slice(start, end))
            index += 1
        nb_patch_per_dim.append((index+1, patch_size[dim] == 1))

    for chunk in itertools.product(*slices):
        patch_slices.append(tuple(chunk))
    
    return patch_slices, nb_patch_per_dim

def getData(data : torch.Tensor, patch_size: list[int], slices: list[slice]) -> list[torch.Tensor]:
    data_sliced = data[[slice(None), slice(None)]+list(slices)]
    padding = []
    for dim_it, _slice in enumerate(reversed(slices)):
        p = 0 if _slice.start+patch_size[-dim_it-1] <= data.shape[-dim_it-1] else patch_size[-dim_it-1]-(data.shape[-dim_it-1]-_slice.start)
        padding.append(0)
        padding.append(p)
    return F.pad(data_sliced, tuple(padding), "constant", -1.5)

class PathCombine():

    def __init__(self) -> None:
        self.data: torch.Tensor = None
        self.overlap: int = None

    def setPatchConfig(self, patch_size: list[int], overlap: int):
        self.data = F.pad(torch.ones([size-overlap*2 for size in patch_size]), [overlap]*2*len(patch_size), mode="constant", value=0)
        self.data = self._setFunction(self.data, overlap)
        dim = len(patch_size)

        A = slice(0, overlap)
        B = slice(-overlap, None)
        C = slice(overlap, -overlap)
        
        for i in range(dim):
            slices_badge = list(itertools.product(*[[A, B] for _ in range(dim-i)]))
            for indexs in itertools.combinations([0,1,2], i):
                result = []
                for slices in slices_badge:
                    slices = list(slices)
                    for index in indexs:
                        slices.insert(index, C)    
                    result.append(tuple(slices))
                for patch, s in zip(PathCombine._normalise([self.data[s] for s in result]), result):
                    self.data[s] = patch


    @staticmethod
    def _normalise(patchs: list[torch.Tensor]) -> list[torch.Tensor]:
        data_sum = torch.sum(torch.concat([patch.unsqueeze(0) for patch in patchs], dim=0), dim=0)
        return [d/data_sum for d in patchs]
            
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.data.repeat([input.shape[0]]+[1]*(len(input.shape)-1)).to(input.device)*input

class Cosinus(PathCombine):

    def __init__(self) -> None:
        super().__init__()

    def _function_sides(self, overlap: int, x: float):
        return np.clip(np.cos(np.pi/(2*(overlap+1))*x), 0, 1)

    def _setFunction(self, data: torch.Tensor, overlap: int) -> torch.Tensor:
        image = sitk.GetImageFromArray(np.asarray(data, dtype=np.uint8))
        danielssonDistanceMapImageFilter = sitk.DanielssonDistanceMapImageFilter()
        distance = torch.tensor(sitk.GetArrayFromImage(danielssonDistanceMapImageFilter.Execute(image)))
        return distance.apply_(partial(self._function_sides, overlap))
    
class Accumulator():

    def __init__(self, patch_slices: list[tuple[slice]], patch_size: list[int], overlap: int, nb_channel: int) -> None:
        self.patchCombine = Cosinus()
        self.patchCombine.setPatchConfig(patch_size, overlap)

        self.patch_slices = []
        for patch in patch_slices:
            slices = []
            for s, shape in zip(patch, patch_size):
                slices.append(slice(s.start, s.start+shape))
            self.patch_slices.append(tuple(slices))
        self.shape = max([[v.stop for v in patch] for patch in patch_slices])
        self.patch_size = patch_size
        self.result = torch.zeros(([nb_channel]+list(max([[v.stop for v in patch] for patch in self.patch_slices])))).to(0)
        
    def addLayer(self, index: int, data: torch.Tensor) -> None:
        slices_dest = tuple([slice(None)] + list(self.patch_slices[index]))
        self.result[slices_dest] += self.patchCombine(data)
    
    def getResult(self):
        return self.result[tuple([slice(None, None)]+[slice(0, s) for s in self.shape])]
