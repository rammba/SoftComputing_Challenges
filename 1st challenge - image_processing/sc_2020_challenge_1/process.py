import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def count_blood_cells(image_path):
    """
    Procedura prima putanju do fotografije i vraca broj crvenih krvnih zrnaca, belih krvnih zrnaca i
    informaciju da li pacijent ima leukemiju ili ne, na osnovu odnosa broja krvnih zrnaca

    Ova procedura se poziva automatski iz main procedure i taj deo kod nije potrebno menjati niti implementirati.

    :param image_path: <String> Putanja do ulazne fotografije.
    :return: <int>  Broj prebrojanih crvenih krvnih zrnaca,
             <int> broj prebrojanih belih krvnih zrnaca,
             <bool> da li pacijent ima leukemniju (True ili False)
    """
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    final, img_final, img_formatted2 = get_white_cells_contours(img_gray, img_rgb)
    rbc, wbc = count_cells(final, img_final, cell="white")
    white_blood_cell_count = wbc
    final, img_final, img_formatted1 = get_red_cells_contours(img_gray, img_rgb, img_formatted2)
    rbc, wbc = count_cells(final, img_final, cell="red")
    red_blood_cell_count = rbc

    if white_blood_cell_count > 0 and red_blood_cell_count / white_blood_cell_count < 11:
        has_leukemia = True
    else:
        has_leukemia = False
    return red_blood_cell_count, white_blood_cell_count, has_leukemia


def get_representative_contours(img_rgb, img_formatted):
    img2, contours, hierarchy = cv2.findContours(img_formatted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img2 = img_rgb.copy()
    circles = {}
    cv2.drawContours(img2, contours, -1, (255, 0, 0), 1)
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        if radius > 10:
            circles[center] = radius
            cv2.circle(img2, center, radius, (0, 255, 0), 2)
    deleting = {}
    for center1 in circles.keys():
        for center2 in circles.keys():
            if center1 != center2:
                distance = np.sqrt(np.square(center2[1] - center1[1]) + np.square(center2[0] - center1[0]))
                if distance < 10:
                    if circles[center1] > circles[center2]:
                        deleting[center2] = circles[center2]
                    else:
                        deleting[center1] = circles[center1]
    final = {}
    for center in circles.keys():
        if center not in deleting.keys():
            final[center] = circles[center]
    img_final = img_rgb.copy()
    return final, img_final


def get_white_cells_contours(img_gray, img_rgb):
    ret, img_threshold = cv2.threshold(img_gray, 143, 255, cv2.THRESH_BINARY)
    img_threshold = 255 - img_threshold
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (4, 4))
    img_formatted = cv2.erode(img_threshold, kernel, iterations=2)
    img_formatted = cv2.dilate(img_formatted, kernel, iterations=3)
    final, img_final = get_representative_contours(img_rgb=img_rgb, img_formatted=img_formatted)
    return final, img_final, img_threshold


def get_red_cells_contours(img_gray, img_rgb, img_white):
    ret, img_threshold = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    img_threshold = 255 - img_threshold
    img_threshold = img_threshold - img_white
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    img_dil = cv2.dilate(img_threshold, kernel, iterations=2)
    img_formatted = cv2.erode(img_dil, kernel, iterations=2)
    final, img_final = get_representative_contours(img_rgb=img_rgb, img_formatted=img_formatted)
    return final, img_final, img_formatted


def count_cells(final, img_final, cell):
    white_blood_cell_count = 0
    red_blood_cell_count = 0
    for center in final.keys():
        cv2.circle(img_final, center, final[center], (0, 255, 0), 2)
        blue_color = 0
        for i in range(center[1] - 10, center[1] + 10):
            for j in range(center[0] - 10, center[0] + 10):
                height, width, channel = img_final.shape
                if i < height and j < width:
                    pixel = img_final[i, j]
                    if pixel[2] > pixel[0] and pixel[2] > pixel[1]:
                        blue_color += 1

        if blue_color > 70 and cell == "white":
            white_blood_cell_count += 1
        elif blue_color <= 70 and cell == "red":
            red_blood_cell_count += 1
    return red_blood_cell_count, white_blood_cell_count
