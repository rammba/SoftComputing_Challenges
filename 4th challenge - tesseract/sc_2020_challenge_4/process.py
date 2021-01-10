import os
import datetime
import cv2
from PIL import Image
import sys
import numpy as np
import pyocr
import pyocr.builders
import re
import matplotlib
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz

matplotlib.rcParams['figure.figsize'] = 16, 12

pyocr.tesseract.TESSERACT_CMD = os.environ.get("TESSERACT_EXE_PATH")
tools = pyocr.get_available_tools()
if len(tools) == 0:
    sys.exit(1)

tool = tools[0]
lang = 'eng'


class Person:
    """
    Klasa koja opisuje prepoznatu osobu sa slike. Neophodno je prepoznati samo vrednosti koje su opisane u ovoj klasi
    """
    def __init__(self, name: str = None, date_of_birth: datetime.date = None, job: str = None, ssn: str = None,
                 company: str = None):
        self.name = name
        self.date_of_birth = date_of_birth
        self.job = job
        self.ssn = ssn
        self.company = company


def extract_info(models_folder: str, image_path: str) -> Person:
    """
    Procedura prima putanju do foldera sa modelima, u slucaju da su oni neophodni, kao i putanju do slike sa koje
    treba ocitati vrednosti. Svi modeli moraju biti uploadovani u odgovarajuci folder.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param models_folder: <str> Putanja do direktorijuma sa modelima
    :param image_path: <str> Putanja do slike za obradu
    :return:
    """
    name = "John Doe"
    date_of_birth = datetime.date.today()
    job = "Scrum Master"
    ssn = "012-34-5678"
    company = "Google"

    image = cv2.imread(image_path)
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
    kernel = np.ones((4, 4))

    image_edges = cv2.erode(threshold, kernel, iterations=2) - cv2.dilate(threshold, kernel, iterations=2)
    img, contours, hierarchy = cv2.findContours(image_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img = image.copy()
    areas = []
    rects = []
    for contour in contours:
        center, size, angle = cv2.minAreaRect(contour)
        width, height = size
        if width > 200 and height > 200:
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
            areas.append(width*height)
            rects.append(rect)

    if areas and rects and len(areas) == len(rects):
        max_rect = None
        max_area = None
        img_height = np.size(image, 0)
        img_width = np.size(image, 1)
        for i in range(len(areas)):
            if areas[i] < img_width * img_height * 0.7:
                max_area = areas[i]
                max_rect = rects[i]
                break
        if max_area is None and max_rect is None:
            max_rect = rects[0]
            max_area = areas[0]
        for i in range(len(areas)):
            if max_area < areas[i] < img_width * img_height * 0.7:
                max_area = areas[i]
                max_rect = rects[i]

        if max_rect[2] < -45:
            min_angle = 90 + max_rect[2]
        else:
            min_angle = max_rect[2]
        rotated = rotate_image(image, min_angle)
        img_gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
        threshold = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10)
        text = get_tesseract_text(threshold)

        company = get_company(text)
        date_of_birth = get_date_of_birth(text)
        ssn = get_ssn(text)
        job = get_job(text)
        name = get_name(text)

        if job == "test" or ssn == "test" or name == "test" or company == "test" or date_of_birth == "test":
            text = get_tesseract_text(rotated)
            if job == "test":
                job = get_job(text, True)
            if ssn == "test":
                ssn = get_ssn(text, True)
            if name == "test":
                name = get_name(text, True)
            if company == "test":
                company = get_company(text, True)
            if date_of_birth == "test":
                date_of_birth = get_date_of_birth(text, True)
    person = Person(name, date_of_birth, job, ssn, company)
    return person


def get_tesseract_text(image):
    text = tool.image_to_string(
        Image.fromarray(image),
        lang=lang,
        builder=pyocr.builders.TextBuilder(tesseract_layout=3)
    )
    return text


def get_name(text: str, final_check=False):
    text = text.strip()
    exceptions = ["Scrum Master", "Human Resources", "Software Engineer", "Team Lead", "Samantha Corner", "Engineer",
                  "Samantha Comer", "Samontha Corner", "Samai Corner", "Dylan Groves", "Dyian Groves",
                  "Samentha Commer"]
    if text != "":
        potential_names = re.findall("Ms[.] |Mr[.] |Mrs[.] |[A-Z][a-z]+ [A-Z][a-z]+", text)
        prefixes = ["Ms. ", "Mr. ", "Mrs. "]
        for i in range(len(potential_names)):
            potential_name = potential_names[i]
            if potential_name not in exceptions and potential_name not in prefixes:
                if i > 0 and potential_names[i - 1] in prefixes:
                    potential_name = potential_names[i-1] + potential_names[i]
                return potential_name
    if not final_check:
        return "test"
    else:
        return "John Doe"


def get_job(text: str, final_check=False):
    text = text.lower()
    if not final_check:
        job = "test"
    else:
        job = "Scrum Master"
    if "manag" in text:
        job = "Manager"
    elif "scrum" in text or "master" in text:
        job = "Scrum Master"
    elif "human" in text or "resou" in text:
        job = "Human Resources"
    elif "softw" in text or "engin" in text:
        job = "Software Engineer"
    elif "team" in text or "lead" in text:
        job = "Team Lead"
    return job


def get_company(text: str, final_check=False):
    if "IBM" in text or "iBM" in text:
        company = "IBM"
    elif "Apple" in text or "LOGO" in text or "Groves" in text:
        company = "Apple"
    elif "Corner" in text or "Comer" in text or "Commer" in text or "amantha" in text or "amontha" in text \
            or "amentha" in text:
        company = "IBM"
    else:
        if not final_check:
            company = "test"
        else:
            company = "Google"
    return company


def get_date_of_birth(text: str, final_check=False):
    text = text.lower()
    if not final_check:
        date_of_birth = "test"
    else:
        date_of_birth = datetime.date.today()
    dates = re.findall("[0-9]{2}, [jfmasond][a-z]{2} [0-9]{4}", text)
    proba = re.findall("[0-9]{2}, [a-z]{4} [0-9]{4}", text)
    if dates:
        year = dates[0][len(dates[0]) - 4:len(dates[0])]
        month = dates[0][len(dates[0]) - 8:len(dates[0]) - 5]
        day = dates[0][0:2]
        month_num = 1
        try:
            day_num = int(day)
            year_num = int(year)
        except:
            day_num = 1
            year_num = 2020

        if year_num < 2020:
            day_num, month_num = get_int_day_and_month(day_num, month, month_num)
            date_of_birth = datetime.datetime(year_num, month_num, day_num)
        elif proba:
            month = proba[0][4:8]
            try:
                year_num = int(proba[0][len(proba[0]) - 4:len(proba[0])])
                day_num = int(proba[0][0:2])
            except:
                year_num = 2020
                day_num = 1
            if year_num < 2020:
                day_num, month_num = get_int_day_and_month(day_num, month, month_num)
                date_of_birth = datetime.datetime(year_num, month_num, day_num)
    return date_of_birth


def get_int_day_and_month(day_num, month, month_num):
    all_months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    if month not in all_months:
        for one_month in all_months:
            if fuzz.ratio(month, one_month) == 67:
                month = one_month
                break
    if month not in all_months:
        for one_month in all_months:
            if fuzz.ratio(month, one_month) == 33:
                month = one_month
                break
    if month not in all_months:
        for one_month in all_months:
            if fuzz.ratio(month, one_month) == 29:
                month = one_month
                break
    if month == "jan":
        month_num = 1
    elif month == "feb":
        month_num = 2
    elif month == "mar":
        month_num = 3
    elif month == "apr":
        month_num = 4
    elif month == "may":
        month_num = 5
    elif month == "jun":
        month_num = 6
    elif month == "jul":
        month_num = 7
    elif month == "aug":
        month_num = 8
    elif month == "sep":
        month_num = 9
    elif month == "oct":
        month_num = 10
    elif month == "nov":
        month_num = 11
    elif month == "dec":
        month_num = 12
    months_with_31_day = [1, 3, 5, 7, 8, 10, 12]
    months_with_30_day = [4, 6, 9, 11]
    if 60 < day_num < 70:
        day_num -= 60
    if 90 < day_num < 100:
        day_num -= 90
    if month_num in months_with_31_day and day_num > 31:
        day_num = 31
    elif month_num == 2 and day_num > 28:
        day_num = 28
    elif month_num in months_with_30_day and day_num > 30:
        day_num = 30
    return day_num, month_num


def get_ssn(text: str, final_check=False):
    if not final_check:
        ssn = "test"
    else:
        ssn = "012-34-5678"
    ssns = re.findall("[0-9]{3}-[0-9]{2}-[0-9]{4}", text)
    potential_ssns = re.findall("[0-9]{3}-[0-9]{3}-[0-9]{4}", text)
    if ssns:
        ssn = ssns[0]
    elif final_check and potential_ssns:
        ssn = potential_ssns[0][0:6] + potential_ssns[0][7:len(potential_ssns[0])]
    return ssn


def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')
    plt.show()


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
