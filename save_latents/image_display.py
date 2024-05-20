from typing import List
import numpy as np
from PyQt5.QtWidgets import QMainWindow, QScrollArea, QLabel, QGridLayout, QVBoxLayout, QStackedLayout, QHBoxLayout, QPushButton, QWidget
from PyQt5.QtGui import QPixmap, QImage
import torch
from sklearn.decomposition import PCA
import json
import math


class GeneratedImage():
    def __init__(self, latent_vector, image_data):
        self.latent_vector = latent_vector
        self.image_data = image_data
        self.label:QLabel = self.create_label(image_data)

    def create_label(self, image_data):
            image_bytes = image_data.to(torch.uint8).numpy().tobytes()

            qimage = QImage(image_bytes, image_data.shape[1], image_data.shape[2], QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimage)
            label = QLabel()
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.setFixedSize(128,128)

            return label
    
class PCADisplay():
    def __init__(self, variation_count, image_data:List[GeneratedImage]=[]):
        self.image_data:List[GeneratedImage] = image_data
        self.variation_count = variation_count

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()

        self.display_layout = QGridLayout()
        self.display_widget = QWidget()

        self.setup_ui()
        self.render(self.image_data)

    def setup_ui(self):        
        
        self.main_widget.setBaseSize(100, 100)
        self.main_widget.setStyleSheet("border: 1px solid #000000;")

        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget.setLayout(self.main_layout)

        self.display_widget.setLayout(self.display_layout)
        self.main_layout.addWidget(self.display_widget,1)

    def render(self, data:List[GeneratedImage]):
        self.arrange_display(self.image_data)

    def arrange_display(self):
        for i, image in enumerate(self.image_data):
            row = i // self.variation_count
            column = i % self.variation_count + 1
            self.display_layout.addWidget(image.label, row, column)
            if(i % self.variation_count == self.variation_count // 2):
                image.label.setStyleSheet("border: 2px solid red;")

class ImageGrid():
    def __init__(self, image_data:List[GeneratedImage]=[]):
        self.selected_images:List[GeneratedImage] = []
        self.image_data:List[GeneratedImage] = image_data.copy()

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()

        self.display_layout = QGridLayout()
        self.display_widget = QWidget()

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.display_widget) 
        self.main_layout.addWidget(self.scroll_area)

        self.button_layout = QHBoxLayout()
        self.button_widget = QWidget()
        
        self.setup_ui()

        self.all_selected_toggle = False

        self.render(self.image_data)

    def setup_ui(self):        
        
        self.main_widget.setBaseSize(800, 400)
        self.main_widget.setStyleSheet("border: 1px solid #000000;")

        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_widget.setLayout(self.main_layout)

        self.display_widget.setLayout(self.display_layout)

        self.button_widget.setLayout(self.button_layout)
        self.main_layout.addWidget(self.button_widget, 0)

        button = QPushButton("Select/Unselect All")
        button.clicked.connect(lambda checked: self.select_all())
        self.add_button(button)

        button = QPushButton("Write to json")
        button.clicked.connect(lambda checked: self.write_selected_latent_vectors())
        self.add_button(button)

    def update_display(self, image_data:List[GeneratedImage]):
        self.clear_layout(self.display_layout)
        self.image_data = image_data.copy()
        self.selected_images = []
        if image_data:
            self.render(self.image_data)

    def render(self, data:List[GeneratedImage]):
        for image in data:
            label = image.label
            label.mousePressEvent = lambda event, label=label: self.on_image_clicked(label)
            label.setStyleSheet("")
        self.arrange_display(self.image_data)

    def arrange_display(self, image_data):
        column_count = math.ceil(len(image_data) / 4)
        for i, image in enumerate(self.image_data):
            row = i // column_count
            column = i % column_count
            self.display_layout.addWidget(image.label, row, column)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0).widget()
            if child:
                layout.removeWidget(child)

    def on_image_clicked(self, label):
        image = None
        for img in self.image_data:
            if img.label == label:
                image = img

        if image in self.selected_images:
            self.highlight_label(label, False)
            self.selected_images.remove(image)
        else:
            self.highlight_label(label, True)
            self.selected_images.append(image)

    def highlight_label(self, label:QLabel, toggle):
        if toggle:
            label.setStyleSheet("border: 2px solid red;")
        else:
            label.setStyleSheet("")

    def add_button(self, button: QPushButton):
        self.button_layout.addWidget(button)

    def get_selected(self):
        return self.selected_images
    
    def select_all(self):
        if not self.all_selected_toggle:
            self.selected_images = [i for i in self.image_data]
            for image in self.selected_images:
                self.highlight_label(image.label, True)
        else:
            self.selected_images = []
            for image in self.image_data:
                self.highlight_label(image.label, False)

        self.all_selected_toggle = not self.all_selected_toggle

    def write_selected_latent_vectors(self):
        selected_latent_vectors = [image.latent_vector.tolist() for image in self.selected_images]
        with open("latents.json", "w") as file:
            json.dump(selected_latent_vectors, file)

class ImageGroups():
    def __init__(self, group_names, image_data, ungroupCallback, G, pc_count):

        # self.G = Generator("faces.pkl")
        self.G = G
        self.pc_count = pc_count
        self.ungroupCallback = ungroupCallback

        self.main_widget = QWidget()
        self.main_layout = QVBoxLayout()
        self.main_widget.setLayout(self.main_layout)

        self.groups_widget = QWidget()
        self.groups_layout = QStackedLayout()
        self.groups_widget.setLayout(self.groups_layout)
        self.main_layout.addWidget(self.groups_widget)

        self.buttons_widget = QWidget()
        self.buttons_layout = QHBoxLayout()
        self.buttons_widget.setLayout(self.buttons_layout)
        self.main_layout.addWidget(self.buttons_widget)

        self.groups = {}
        self.group_names = group_names
        self.image_data = image_data

        self.setup_groups(self.group_names)
        self.setup_buttons()

        self.pca_windows = []


    def setup_groups(self, group_names):
        colors = ['red', 'green', 'blue']
        for i, group_name in enumerate(group_names):
            group = ImageGrid([])
            group.display_widget.setStyleSheet(f"background-color: {colors[i]}")
            self.groups_layout.addWidget(group.main_widget)

            for button in self.get_group_buttons(group_name):
                group.add_button(button)

            self.groups[group_name] = group

    def get_group_buttons(self, group_name):
        ungroup_button = QPushButton("ungroup selected")
        ungroup_button.clicked.connect(lambda checked: self.ungroupCallback(group_name))
        # pca
        pca_button = QPushButton("PCA")
        pca_button.clicked.connect(lambda checked: self.run_pca(group_name))

        return [ungroup_button, pca_button]
    
    def run_pca(self, group_name):
        group:ImageGrid = self.get_group(group_name)
        latent_vectors = [image.latent_vector for image in group.image_data]
        pca = PCA(n_components=self.pc_count)
        pca.fit(latent_vectors)
        print("fitting pca")
        components_list = pca.components_.tolist()
        named_pcs = {}
        for i, pc in enumerate(components_list):
            named_pcs[f"pc_{i}"] = pc
        named_pcs['group_mean'] = np.mean(latent_vectors, axis=0).tolist()
        named_pcs['samples'] = [lv.tolist() for lv in latent_vectors]

        with open(f"{group_name}_components.json", "w") as file:
            json.dump(named_pcs, file)

        view_samples = latent_vectors[:5]
        view_samples.append(np.mean(latent_vectors, axis=0))

        for i, pc in enumerate(pca.components_):
            print(f"generating pc {i} preview")
            self.view_pca(pc, view_samples, f"principal_compnent_{i}")

    def view_pca(self, principal_component, initial_latents, component_id):

        def load_images(latent_vectors):
            images = []
            for w in latent_vectors:
                images.append(GeneratedImage(w, self.G.generate(w.reshape(1, -1))))
            return images
        
        variation_count = 7
        increments = np.linspace(-10, 10, variation_count).tolist()

        initial_latents_copy = initial_latents

        latents = []
        for w in initial_latents_copy:
            for alpha in increments:
                latents.append(w + principal_component * alpha)

        images = load_images(latents)

        # Create a new ImageDisplay with the images
        image_display = PCADisplay(variation_count, images)

        # Create a new QMainWindow
        pca_window = QMainWindow()
        pca_window.setWindowTitle(f"pc_{component_id}")
        pca_window.setCentralWidget(image_display.main_widget)
        pca_window.show()

        self.pca_windows.append(pca_window)

    def display_group(self, group_name):
        group_index = self.group_names.index(group_name)
        self.groups_layout.setCurrentIndex(group_index)

    def setup_buttons(self):
        for group_name in self.group_names:
            button = QPushButton(group_name)
            button.clicked.connect(lambda checked, name=group_name: self.display_group(name))
            self.buttons_layout.addWidget(button)

    def update_group(self, group_name, image_data):
        group:ImageGrid = self.groups[group_name]
        group.update_display(group.image_data.extend(image_data))

    def get_group(self, group_name):
        return self.groups[group_name]
