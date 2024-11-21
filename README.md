
from .utils import divide_tiff_into_pages
import easyocr
from celery import Celery
from celery import group, chord, Task
from .config import Config
import numpy as np
import base64
import cv2
from easyocr.utils import reformat_input
from easyocr.recognition import get_text
import torch
torch.set_num_threads(1)

celery = Celery(__name__)
celery.conf.broker_url = Config.CELERY_BROKER_URL
celery.conf.result_backend = Config.CELERY_RESULT_BACKEND


class EasyOCRReaderTask(Task):
    reader = None  # Cache the reader per worker process

    def __call__(self, *args, **kwargs):
        if self.reader is None:
            # Initialize the EasyOCR Reader once per worker process
            self.reader = easyocr.Reader(['en'], gpu=False)
        return super().__call__(*args, **kwargs)


@celery.task(name='tasks.OCR_task')
def OCR_task(tiff_data, task_id):
    """
    Orchestrates OCR for a multi-page TIFF document.
    Args:
        tiff_data: The TIFF file data.
        task_id: Task ID for tracking.
    Returns:
        Aggregated OCR results for all pages.
    """
    # Step 1: Divide the TIFF file into pages
    pages = divide_tiff_into_pages(tiff_data)

    # Step 2: Trigger detection tasks for all pages using a chord
    detection_tasks = group(
        detect_task.s(page).set(queue="detect_queue") for page in pages
    )
    final_result = chord(detection_tasks)(
        aggregate_page_results.s().set(queue="aggregate_page_queue", task_id=task_id)
    )

    # Return an async result that can be tracked
    return final_result


@celery.task(name='tasks.detect', base=EasyOCRReaderTask)
def detect_task(page_data):
    """
    Detect text regions in the image and trigger recognition tasks for each region.
    Args:
        page_data: Image data for the page.
    Returns:
        Aggregated recognized text for the page.
    """
    reader = detect_task.reader
    if reader.recognizer is not None:
        del detect_task.reader.recognizer
        detect_task.reader.recognizer = None
    img, img_cv_grey = reformat_input(page_data)

    # Perform detection
    horizontal_list, free_list = reader.detect(
        img,
        min_size=20,
        text_threshold=0.7,
        low_text=0.4,
        link_threshold=0.4,
        canvas_size=2560,
        mag_ratio=1.0,
        slope_ths=0.1,
        ycenter_ths=0.5,
        height_ths=0.5,
        width_ths=0.5,
        add_margin=0.1,
        threshold=0.2,
        bbox_min_score=0.2,
        bbox_min_size=3,
        max_candidates=0,
    )

    result = []

    to_read = []

    for bbox in horizontal_list[0]:
        h_list = [bbox]
        f_list = []
        image_list, max_width = easyocr.utils.get_image_list(h_list, f_list, img_cv_grey, model_height=64)
        to_read.append([image_list, max_width])

    to_read_base64 = []
    for image_list, max_width in to_read:
        image_list_base64 = []
        boxes = []
        for bbox, crop_img in image_list:
            _, buffer = cv2.imencode('.png', crop_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            image_list_base64.append(img_base64)
            boxes.append(bbox)
        boxes = [[[int(coord[0]), int(coord[1])] for coord in region] for region in boxes]
        to_read_base64.append((image_list_base64, boxes, max_width))

    # Flatten and combine detected regions
    # print(horizontal_list)
    # print(free_list)
    all_regions = horizontal_list[0]  #+ free_list[0]

    # Encode the image to Base64 for transmission

    # Convert numpy types to plain Python types
    all_regions = [list(map(int, region)) for region in all_regions]

    # Trigger recognition tasks for all regions using a chord

    recognition_tasks = group(
        recognize_task.s({"image_list_base64": image_list, "boxes": boxes, "max_width": max_width}).set(queue="recognize_queue")
        for image_list, boxes, max_width in to_read_base64
    )
    return chord(recognition_tasks)(aggregate_document_results.s().set(queue="aggregate_document_queue"))


@celery.task(name='tasks.recognize', base=EasyOCRReaderTask)
def recognize_task(data):
    """
    Recognize text from a single detected region.
    Args:
        data: Contains a single region and the Base64-encoded image.
    Returns:
        Recognized text for the region.
    """
    reader = recognize_task.reader
    if reader.detector is not None:
        del reader.detector
        reader.detector = None

    # Decode the Base64 image
    image_list_base64 = data['image_list_base64']
    max_width = data['max_width']
    boxes = data['boxes']

    image_list = []
    for img_base64, box in zip(image_list_base64,boxes):
        img_data = base64.b64decode(img_base64)
        img_cv_grey = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        image_list.append((box, img_cv_grey))

    # Perform recognition for the specific region
    # region = [data['region']]  # Wrap in a list since recognize expects a list
    # result = reader.recognize(
    #     img_cv_grey,
    #     horizontal_list=region,
    #     free_list=[],
    #     decoder='greedy',
    #     beamWidth=5,
    #     batch_size=1,
    #     workers=0,
    #     allowlist=None,
    #     blocklist=None,
    #     detail=0,
    #     rotation_info=None,
    #     paragraph=False,
    #     contrast_ths=0.1,
    #     adjust_contrast=0.5,
    #     filter_ths=0.003,
    #     y_ths=0.5,
    #     x_ths=1.0,
    #     output_format="standard"
    # )

    ignore_char = ''.join(set(recognize_task.reader.character) - set(recognize_task.reader.lang_char))
    print(ignore_char)
    result0 = get_text(recognize_task.reader.character, 64, int(max_width), recognize_task.reader.recognizer,
                       recognize_task.reader.converter, image_list,
                       ignore_char, "greedy", 5, 1, 0.1, 0.5, 0.003,
                       0, 'cpu')

    # Return recognized text
    print(result0)
    return " ".join([item[1] for item in result0])


@celery.task(name='tasks.aggregate_page_results')
def aggregate_page_results(results):
    print(results[0][0][0])
    return results


@celery.task(name='tasks.aggregate_document_results')
def aggregate_document_results(results):
    print(results)
    return results
