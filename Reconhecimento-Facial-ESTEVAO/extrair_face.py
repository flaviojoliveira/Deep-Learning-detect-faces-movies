#rodar com pyton 3.8(base)
from PIL import Image
from mtcnn import MTCNN
from os import listdir
from os.path import isdir
from numpy import asarray

detector = MTCNN()


def extrair_face(arquivo, size=(160, 160)):
    img = Image.open(arquivo)
    img = img.convert('RGB')
    array = asarray(img)

    results = detector.detect_faces(array)

    x1, y1, width, height = results[0]['box']

    x2 = x1 + width
    y2 = y1 + height

    face = array[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(size)
    return image


def load_fotos(directory_src, directory_targe):
    for filename in listdir(directory_src):
        path = directory_src + filename
        path_tg = directory_targe + filename

        try:
            face = extrair_face(path)
            print(face)
            face.save(path_tg, "JPEG", quality=100, optimize=True, progressive=True)
        except:
            print("Erro na imagem{}".format(path))


def load_dir(directory_src, directory_target):

    for subdir in listdir(directory_src):
        path = directory_src + subdir + "\\"

        path_tg = directory_target + subdir + "\\"
        print('antes')
        print(path, path_tg)
        if not isdir(path):
            continue

        print(path, path_tg)
        load_fotos(path, path_tg)


if __name__ == '__main__':
    load_dir("C:\\Users\mayki\\PycharmProjects\\pythonProject1\\entradas_fotos_treinamento\\", #Pasta contendo as imagens que serão processadas
             "C:\\Users\\mayki\\PycharmProjects\\pythonProject1\\validacao_faces\\") # pastas onde as imagens processadas são guardadas
