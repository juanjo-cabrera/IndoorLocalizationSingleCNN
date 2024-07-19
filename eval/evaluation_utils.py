
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch.nn.functional as F
import torch
from operator import itemgetter
import time
from config import PARAMS


class FreiburgMap():
    def __init__(self, map_dset, model, activation):
        self.map_dset = map_dset
        self.get_whole_map()
        self.compute_whole_vectors(model, activation)
    def get_coordinates(self, imgs_tuple):
        map_coordinates = []
        for img_tuple in imgs_tuple:
            ruta = img_tuple[0]
            x_index = ruta.index('_x')
            y_index = ruta.index('_y')
            a_index = ruta.index('_a')
            x=ruta[x_index+2:y_index]
            y=ruta[y_index+2:a_index]
            coor_list= [x,y]
            coor = torch.from_numpy(np.array(coor_list,dtype=np.float32))
            map_coordinates.append(coor)
        return map_coordinates
    def get_room_map(self, which_room):
        room_imgs_idxs = np.where(np.array(self.map_dset.targets) == which_room)[0]
        room_imgs_root = itemgetter(*room_imgs_idxs)(self.map_dset.imgs)
        room_coordinates = self.get_coordinates(room_imgs_root)
        room_imgs = itemgetter(*room_imgs_idxs)(self.map_dset)
        return room_imgs, room_coordinates
    def get_whole_map(self):
        self.room0_imgs, self.room0_coordinates = self.get_room_map(0)
        self.room1_imgs, self.room1_coordinates = self.get_room_map(1)
        self.room2_imgs, self.room2_coordinates = self.get_room_map(2)
        self.room3_imgs, self.room3_coordinates = self.get_room_map(3)
        self.room4_imgs, self.room4_coordinates = self.get_room_map(4)
        self.room5_imgs, self.room5_coordinates = self.get_room_map(5)
        self.room6_imgs, self.room6_coordinates = self.get_room_map(6)
        self.room7_imgs, self.room7_coordinates = self.get_room_map(7)
        self.room8_imgs, self.room8_coordinates = self.get_room_map(8)

    def load_room_map(self, room_number):
        if room_number == 0:
            room_imgs, room_coordinates = self.room0_imgs, self.room0_coordinates
        elif room_number == 1:
            room_imgs, room_coordinates = self.room1_imgs, self.room1_coordinates
        elif room_number == 2:
            room_imgs, room_coordinates = self.room2_imgs, self.room2_coordinates
        elif room_number == 3:
            room_imgs, room_coordinates = self.room3_imgs, self.room3_coordinates
        elif room_number == 4:
            room_imgs, room_coordinates = self.room4_imgs, self.room4_coordinates
        elif room_number == 5:
            room_imgs, room_coordinates = self.room5_imgs, self.room5_coordinates
        elif room_number == 6:
            room_imgs, room_coordinates = self.room6_imgs, self.room6_coordinates
        elif room_number == 7:
            room_imgs, room_coordinates = self.room7_imgs, self.room7_coordinates
        elif room_number == 8:
            room_imgs, room_coordinates = self.room8_imgs, self.room8_coordinates

        return room_imgs, room_coordinates

    def get_latent_vector(self, image, model, activation):
        image = image.cuda().unsqueeze(0)
        output = model(image)
        latent_vector = activation['latent_vector'].flatten()
        return latent_vector

    def compute_room_vectors(self, room_number, model, activation):
        room_vectors = []
        room_imgs, room_coordinates = self.load_room_map(room_number)
        for room_img in room_imgs:
            img = room_img[0]
            latent_vector = self.get_latent_vector(img, model, activation)
            room_vectors.append(latent_vector)
        return room_vectors

    def compute_whole_vectors(self, model, activation):
        self.room0_vectors = self.compute_room_vectors(0, model, activation)
        self.room1_vectors = self.compute_room_vectors(1, model, activation)
        self.room2_vectors = self.compute_room_vectors(2, model, activation)
        self.room3_vectors = self.compute_room_vectors(3, model, activation)
        self.room4_vectors = self.compute_room_vectors(4, model, activation)
        self.room5_vectors = self.compute_room_vectors(5, model, activation)
        self.room6_vectors = self.compute_room_vectors(6, model, activation)
        self.room7_vectors = self.compute_room_vectors(7, model, activation)
        self.room8_vectors = self.compute_room_vectors(8, model, activation)

    def load_room_vectors(self, room_number):
        if room_number == 0:
            room_vectors = self.room0_vectors
        elif room_number == 1:
            room_vectors = self.room1_vectors
        elif room_number == 2:
            room_vectors = self.room2_vectors
        elif room_number == 3:
            room_vectors = self.room3_vectors
        elif room_number == 4:
            room_vectors = self.room4_vectors
        elif room_number == 5:
            room_vectors = self.room5_vectors
        elif room_number == 6:
            room_vectors = self.room6_vectors
        elif room_number == 7:
            room_vectors = self.room7_vectors
        elif room_number == 8:
            room_vectors = self.room8_vectors

        return room_vectors

    def evaluate_error_position(self, test_img, coor_test, model, activation):
        start_time = time.time()
        output = model(test_img)
        test_vector = activation['latent_vector'].flatten()
        # print(test_vector.shape[0])
        _, room_predicted = torch.max(output.data, 1)
        room_vectors = self.load_room_vectors(room_predicted)
        room_images, room_coors = self.load_room_map(room_predicted)
        distances = []
        for vector in room_vectors:
            # print('Vector size: ', vector.shape[0])
            # print(f'Memory size of a vector: {vector.element_size() * vector.nelement()} Bytes')
            euclidean_distance = F.pairwise_distance(test_vector, vector, keepdim=True)
            distances.append(euclidean_distance)
        ind_min = distances.index(min(distances))

        coor_map = room_coors[ind_min]
        end_time = time.time()
        processing_time = end_time - start_time
        # print(f'Processing time: {processing_time}')
        error_localizacion = F.pairwise_distance(coor_test, coor_map.cuda())
        return error_localizacion.detach().cpu().numpy(), room_predicted, processing_time


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()


map_data = dset.ImageFolder(root=PARAMS.map_dir, transform=transforms.ToTensor())
map_dataloader = DataLoader(map_data,
                        shuffle=False,
                        num_workers=0,
                        batch_size=1)

