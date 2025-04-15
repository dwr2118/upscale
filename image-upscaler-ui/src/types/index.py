from typing import List, Dict, Any

class ImageUpload:
    def __init__(self, filename: str, filetype: str, filesize: int):
        self.filename = filename
        self.filetype = filetype
        self.filesize = filesize

class UpscaledImage:
    def __init__(self, original_filename: str, upscaled_filename: str, dimensions: Dict[str, int]):
        self.original_filename = original_filename
        self.upscaled_filename = upscaled_filename
        self.dimensions = dimensions

class ProcessingResult:
    def __init__(self, success: bool, message: str, upscaled_images: List[UpscaledImage] = None):
        self.success = success
        self.message = message
        self.upscaled_images = upscaled_images if upscaled_images is not None else []