from __future__ import print_function
import cv2
import numpy as np
import matplotlib.pyplot as plt
import collections
from sklearn import datasets
from sklearn.cluster import KMeans
from fuzzywuzzy import fuzz

from keras.models import Sequential
from keras.layers.core import Dense,Activation
from keras.optimizers import SGD
from keras.models import model_from_json

import matplotlib.pylab as pylab
pylab.rcParams['figure.figsize'] = 16, 12


def train_or_load_character_recognition_model(train_image_paths, serialization_folder):
    """
    Procedura prima putanje do fotografija za obucavanje (dataset se sastoji iz razlicitih fotografija alfabeta), kao i
    putanju do foldera u koji treba sacuvati model nakon sto se istrenira (da ne trenirate svaki put iznova)

    Procedura treba da istrenira model i da ga sacuva u folder "serialization_folder" pod proizvoljnim nazivom

    Kada se procedura pozove, ona treba da trenira model ako on nije istraniran, ili da ga samo ucita ako je prethodno
    istreniran i ako se nalazi u folderu za serijalizaciju

    :param train_image_paths: putanje do fotografija alfabeta
    :param serialization_folder: folder u koji treba sacuvati serijalizovani model
    :return: Objekat modela
    """
    ann = load_trained_ann()
    if ann is None:
        letters = []
        for img_path in train_image_paths:
            img_rgb = load_image(img_path)
            img_binary = invert(image_bin(image_gray(img_rgb)))
            img_bin = erode(dilate(img_binary))
            selected_regions, let, region_distances = select_roi(img_rgb.copy(), img_binary)
            for letter in let:
                letters.append(letter)
        alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h',
                    'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']
        inputs = prepare_for_ann(letters)
        outputs = convert_output(alphabet)
        print("Traniranje modela zapoceto.")
        ann = create_ann()
        ann = train_ann(ann, inputs, outputs)
        print("Treniranje modela zavrseno.")
        serialize_ann(ann)

    model = ann
    return model


def extract_text_from_image(trained_model, image_path, vocabulary):
    """
    Procedura prima objekat istreniranog modela za prepoznavanje znakova (karaktera), putanju do fotografije na kojoj
    se nalazi tekst za ekstrakciju i recnik svih poznatih reci koje se mogu naci na fotografiji.
    Procedura treba da ucita fotografiju sa prosledjene putanje, i da sa nje izvuce sav tekst koriscenjem
    openCV (detekcija karaktera) i prethodno istreniranog modela (prepoznavanje karaktera), i da vrati procitani tekst
    kao string.

    Ova procedura se poziva automatski iz main procedure pa nema potrebe dodavati njen poziv u main.py

    :param trained_model: <Model> Istrenirani model za prepoznavanje karaktera
    :param image_path: <String> Putanja do fotografije sa koje treba procitati tekst.
    :param vocabulary: <Dict> Recnik SVIH poznatih reci i ucestalost njihovog pojavljivanja u tekstu
    :return: <String>  Tekst procitan sa ulazne slike
    """
    img_rgb = load_image(image_path)
    img_binary = invert(image_bin(image_gray(img_rgb)))
    selected_regions, letters, region_distances = select_roi(img_rgb.copy(), img_binary)
    region_distances = np.array(region_distances).reshape(len(region_distances), 1)

    k_means = KMeans(n_clusters=2, max_iter=2000, tol=0.00001, n_init=10)
    k_means.fit(region_distances)
    inputs = prepare_for_ann(letters)
    results = trained_model.predict(np.array(inputs, np.float32))
    alphabet = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
                'Š', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ž', 'a', 'b', 'c', 'č', 'ć', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 'š', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ž']

    extracted_text = display_result(results, alphabet, k_means)
    return split_text(extracted_text, vocabulary)


def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, img_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_OTSU)
    return img_bin


def invert(image):
    return 255-image


def display_image(image, color=False):
    if color:
        plt.imshow(image)
        plt.show()
    else:
        plt.imshow(image, 'gray')
        plt.show()


def dilate(image):
    kernel = np.ones((3, 3))
    return cv2.dilate(image, kernel, iterations=1)


def erode(image):
    kernel = np.ones((3, 3))
    return cv2.erode(image, kernel, iterations=1)


def resize_region(region):
    '''Transformisati selektovani region na sliku dimenzija 28x28'''
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    all_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 100:
            all_regions.append(Rectangle(x, y, h, w))
    deleting_regions = []
    for r1 in all_regions:
        for r2 in all_regions:
            if r1 != r2 and is_first_rect_in_second(r1, r2):
                deleting_regions.append(r1)
    final_regions = []
    for region in all_regions:
        if region not in deleting_regions:
            final_regions.append(region)
    carons = []
    letters_with_carons = []
    for r1 in final_regions:
        for r2 in final_regions:
            if r1 != r2 and r1.x > r2.x and r1.x+r1.width <= r2.x + r2.width+10 and r1.y < r2.y:
                carons.append(r1)
                letters_with_carons.append(r2)
    final_regions_carons = []
    for region in final_regions:
        if region not in carons and region not in letters_with_carons:
            final_regions_carons.append(region)
    for region in merge_letters_with_carons(letters_with_carons, carons):
        final_regions_carons.append(region)
    for r in final_regions_carons:
        region = image_bin[r.y:r.y + r.height + 1, r.x:r.x + r.width + 1]
        regions_array.append([resize_region(region), (r.x, r.y, r.width, r.width)])
        cv2.rectangle(image_orig, (r.x, r.y), (r.x + r.width, r.y + r.height), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []
    for index in range(0, len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)
    return image_orig, sorted_regions, region_distances


def scale_to_range(image): # skalira elemente slike na opseg od 0 do 1
    ''' Elementi matrice image su vrednosti 0 ili 255.
        Potrebno je skalirati sve elemente matrica na opseg od 0 do 1
    '''
    return image/255


def matrix_to_vector(image):
    '''Sliku koja je zapravo matrica 28x28 transformisati u vektor sa 784 elementa'''
    return image.flatten()


def prepare_for_ann(regions):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann():
    '''
    Implementirati veštačku neuronsku mrežu sa 28x28 ulaznih neurona i jednim skrivenim slojem od 128 neurona.
    Odrediti broj izlaznih neurona. Aktivaciona funkcija je sigmoid.
    '''
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(60, activation='sigmoid'))
    return ann


def train_ann(ann, x_train, y_train):
    x_train = np.array(x_train, np.float32)
    y_train = np.array(y_train, np.float32)
    sgd = SGD(lr=0.11, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(x_train, y_train, epochs=500, batch_size=1, verbose=0, shuffle=False)

    return ann


def serialize_ann(ann):
    model_json = ann.to_json()
    with open("serialized_model/neuronska.json", "w") as json_file:
        json_file.write(model_json)
    ann.save_weights("serialized_model/neuronska.h5")


def load_trained_ann():
    try:
        json_file = open('serialized_model/neuronska.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        ann = model_from_json(loaded_model_json)
        ann.load_weights("serialized_model/neuronska.h5")
        print("Istrenirani model uspesno ucitan.")
        return ann
    except Exception as e:
        return None


def winner(output):
    '''pronaći i vratiti indeks neurona koji je najviše pobuđen'''
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet, k_means):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    w_space_group = max(enumerate(k_means.cluster_centers_), key=lambda x: x[1])[0]
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if k_means.labels_[idx] == w_space_group:
            result += ' '
        result += alphabet[winner(output)]
    return result


def split_text(text: str, vocabulary):
    ret = ""
    words = text.split()
    for word in words:
        percent = np.rint((1 / len(word)) * 100)
        if word not in vocabulary.keys():
            if len(word) == 1:
                ret += "I "
                continue
            different_letters = 0
            for w in vocabulary.keys():
                if fuzz.ratio(word, w) == 100-percent and len(word) > 1:
                    ret += w + " "
                    different_letters = 1
                    break
            if different_letters == 0:
                for w in vocabulary.keys():
                    if fuzz.ratio(word, w) == 100 - percent - percent and len(word) > 2:
                        ret += w + " "
                        different_letters = 2
                        break
            if different_letters == 0:
                for w in vocabulary.keys():
                    if fuzz.ratio(word, w) == 100 - percent - percent - percent and len(word) > 3:
                        ret += w + " "
                        different_letters = 3
                        break
        else:
            ret += word + " "
    return ret.strip(" ")


def is_first_rect_in_second(rect1, rect2):
    if rect1.x > rect2.x and rect1.x+rect1.width < rect2.x+rect2.width and rect1.y > rect2.y \
            and rect1.y+rect1.height < rect2.y+rect2.height:
        return True
    return False


def merge_letters_with_carons(letters, carons):
    final_letters = []
    for i in range(0, len(letters)):
        final_letters.append(Rectangle(letters[i].x, carons[i].y, letters[i].height+(letters[i].y-carons[i].y), letters[i].width))
    return final_letters


class Rectangle:

  def __init__(self, x, y, height, width):
    self.x = x
    self.y = y
    self.height = height
    self.width = width
